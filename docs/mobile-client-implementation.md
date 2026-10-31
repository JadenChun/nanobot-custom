# Mobile client: implementation steps for every task

This is the detailed companion to [the complete backlog](mobile-client-tasks.md).
Task IDs and anchors are stable. No commands below were run during planning; proposed files,
scripts, routes and platform projects do not exist yet. The [accepted plan](mobile-client-plan.md)
defines protocol/security behavior; this guide decomposes it without changing the architecture.
Implementation starts when the user invokes the backlog's OpenCode prompt; this planning task prepares it only.

## Common execution and evidence contract

Before each task, verify its prerequisites and current source, record owned paths and preserve others'
edits. Capture a failing behavior test before implementation, implement minimally, refactor, then run
green focused tests. For toolchain/device/network work use a frozen deterministic pre-check and
post-check instead. Keep the failed output; a missing test dependency is not the intended red result.
Do not weaken assertions, add suppressions/skips, silently refresh baselines or run `ruff format`.

Use an ignored, access-controlled `mobile/evidence/<run-id>/Mxx/` directory (proposed) for sanitized
logs, manifests, test results and screenshots; keep private setup/secrets outside the repository and
shared evidence. Each task report contains objective, actual base/head, diff/owned paths, source and
dependency SHA-256 manifests, command/cwd/exit codes, red/green results, criterion PASS/FAIL/PENDING,
artifact index, cleanup and residual inputs. A gitignored file is not automatically safe to share.
Review the actual final files/artifacts, not only task summaries. Any later code, dependency, config or
artifact change invalidates affected checks and prior review: rerun them and publish a new final manifest.

### Existing commands and their scope

Run from this mobile worktree with its own environments. These commands exist in the inspected
repository; test names must be refreshed at M10 if committed staging changes rename their seams.

```bash
# Read-only implementation-start preflight, repository root
pwd
git branch --show-current
git rev-parse HEAD
git status --short
git diff --name-only

# Environment preparation, implementation only; never reuse staging's editable environment
uv sync --all-extras --dev
uv run --no-sync python -m scripts.install_channel_dependencies --all-channels

# Focused production security/lifecycle protection, repository root
uv run --no-sync python -m pytest tests/webui/test_http_utils.py tests/webui/test_webui_mutation_replay_auth.py tests/webui/test_managed_media_access.py tests/webui/test_attachment_ingress.py tests/webui/test_permission_v2_api.py tests/security/test_native_admission.py tests/security/test_permission_v2_authority.py tests/desktop/test_adapter.py tests/gateway/test_runtime.py tests/gateway/test_gateway_service.py
ruff check nanobot/

# Existing focused WebUI checks, run from webui/
bun run test -- src/tests/bootstrap.test.ts src/tests/runtime.test.ts src/tests/nanobot-client.test.ts src/tests/thread-shell.test.tsx src/tests/sw.test.ts src/tests/main-pwa-registration.test.tsx
bun run lint
bun run build
bun run test

# Required final local Python run; do not substitute CI platform exclusions
uv run --no-sync python -m pytest
uv run --no-sync basedpyright --outputjson
python scripts/check_type_regressions.py BEFORE.json AFTER.json --baseline-root BEFORE_SOURCE --current-root AFTER_SOURCE
```

`BEFORE.json`, `AFTER.json`, `BEFORE_SOURCE` and `AFTER_SOURCE` above are recorded immutable artifacts,
not literal repository paths. Freeze full-source checker reports/source snapshots before the first
production Python task and after each bounded batch under identical dependency/configuration hashes.
Capture checker exits even when nonzero; compare diagnostic signatures across **all** files, not only
the changed scope. The comparator attests diagnostics only: retain input hashes and environment
evidence separately. See [migration attribution](migration/p6-error-attribution.md); historical zero
counts do not attest this checkout. New diagnostic signatures fail the batch. Existing debt is reported,
not absorbed into a new baseline or silently repaired outside ownership. Full-green/final stage claims
require unchanged strict exit zero and the required full local suite passing; a bounded batch PASS
does not close migration P6. P0 fixture-only tasks need their own checks, not unrelated full Python suites.

### Proposed checks to implement and document

The following are **proposed interfaces**, not working scripts today. M01/M02 create the fixture/toolchain
commands; M05/M14 extend the emulator runner; M21 creates the device runner. Record the final real CLI
and underlying commands if a compatible implementation chooses a different invocation.

```bash
# Proposed isolated fixture environment, never the root or staging venv
uv run --project mobile/fixture --frozen python -m pytest
# Proposed module/project locations; standard tool commands after scaffold exists
cd mobile/native-network
go test ./...
# From repository root; generated Gradle wrapper/tasks must be verified in M01
mobile/android/gradlew -p mobile/android :app:testDebugUnitTest :app:connectedDebugAndroidTest :app:assembleDebug
# Proposed qualification runners, from root; each records commands, environment and cleanup
mobile/scripts/qualify-emulator android --phase fixture --evidence-dir <private-run-directory>
mobile/scripts/qualify-emulator android --phase backend --evidence-dir <private-run-directory>
mobile/scripts/qualify-device android --evidence-dir <private-run-directory>
# Proposed backend tests as their files land, using the root's isolated environment
uv run --no-sync python -m pytest tests/remote
# Proposed P1 local staging CLI: finalize help/selectors/dry-run before any live invocation
uv run --no-sync python -m nanobot.remote.staging --help
```

The guide uses **fixture checks**, **native checks**, **backend checks**, **WebUI checks** and **full gates**
to refer to those command groups. Always expand them into actual commands and exit codes in reports.
Gradle and Go successes are labeled by ABI and execution versus compilation. Mocked HTTP/IPC/policy
tests never satisfy real network, OS custody, service or device criteria.

<a id="ownership"></a>
## Ownership ledger and non-goals

- M01–M05: one P0 writer, **new `mobile/` only**, including local lockfiles/ignore rules/scripts.
  No `webui/`, `nanobot/`, root config/dependency files, installed app or staging runtime edits.
- M06–M13: one backend writer for P1/P2, with M10 a serialized integration checkpoint.
  M06's `mobile/contracts/` is handed over after P0; no concurrent mobile edits there.
  P1 retains its listener/approval/entry modules through M15 real-backend qualification.
- M14–M15: one native writer, only `mobile/`; coordinate shared contract changes with backend owner.
  M16–M17: one shared WebUI/build writer. No implicit desktop/native capability inheritance.
- M18–M19: explicit transfer of named remote/config/settings/shared files from backend and P4 owners;
  old writers stop first. M20–M22 are verification/release owners; code defects return to their owner.
- M23–M29: not dispatched. Their listed files are proposed future boundaries, admitted only by a
  separately scoped task. Concurrent changes to the same module/lockfile/generated output are forbidden.

No task grants arbitrary URL/dial/LocalAPI/shell access from the bridge, privileged remote HTML,
offline private history/write queues, temporary mobile chats, a second agent/runtime, phone model keys,
automatic cloud fallback, global Full, new public ingress or home filesystem/tool access through a model.
No task authorizes merge, production activation, store publication or an invented host/recipient/credential.
SDK/version/support-floor choices are measured and recorded; no maintained Capacitor Tailscale SDK is assumed.

<a id="m01"></a>
## M01 — Isolated Android shell, toolchain and evidence harness (P0)

**Objective/prerequisites:** start from the clean mobile worktree; establish reproducible development inputs.
**Allowed:** proposed `mobile/{package.json,bun.lock,capacitor.config.ts,tsconfig.json,vite.config.ts,.gitignore,README.md}`,
`mobile/src/fixture/`, generated `mobile/android/`, `mobile/scripts/`, `mobile/toolchain/` and ignored evidence.
**Forbidden:** all production/root dependency files and the installed staging environment; retain P0-only ownership.

1. Run preflight; record actual clean HEAD, baseline ancestry, source hashes and absence of a mobile project.
   Accept a later docs-only handoff commit. Stop on unrelated dirty files without stashing/resetting them.
2. Choose and pin compatible Capacitor/React/Go/JDK/SDK/NDK/Gradle/plugin versions and provenance/licenses
   using the accepted official references. Record actual OS/API/WebView/ABI support, not guessed floors.
3. Scaffold the smallest packaged fixture page and Android plugin seam under `mobile/`; keep mobile asset
   output at `mobile/www`. Isolate Python/JS/Go caches and SDK/AVD paths from staging's environments.
4. Create an x86_64 AVD; measure acceleration, boot, WebView and native execution. Add separate arm64-v8a
   compile targets. Record SDK license/tool availability honestly; do not present compilation as a device run.
5. Add evidence/qualification scaffolding with finite timeouts and cleanup; pin dependencies/locks and
   exclude keys/node state/APKs/private evidence from source. Rebuild from the recorded inputs.

**Checks:** deterministic before/after source/lock manifests; generated Gradle wrapper build and a tiny native
ABI probe; repeat build from clean outputs. Negative: missing tool/ABI/acceleration produces a named failure.
**Gate/evidence:** x86_64 boots and executes the probe; arm64 separately compiles; toolchain manifest and
command exits identify reproducible inputs. No transport or hardware-backed security claim yet.
**Rollback/inputs:** remove only task-created AVD/build outputs/environments; leave host services untouched.
Unavailable SDK components/licenses block their criterion; fixture contract drafting can continue locally.

<a id="m02"></a>
## M02 — Non-agent protocol and media fixture (P0)

**Objective/prerequisites:** [M01](#m01); create deterministic peers without starting another Nanobot agent.
**Allowed:** proposed `mobile/fixture/{pyproject.toml,uv.lock,app.py,tests/}`, `mobile/contracts/fixture-v1.json`,
`mobile/src/fixture/` and fixture-only script extensions. **Forbidden:** production routes/config/data/model calls.

1. Define fixed fixture operations for bootstrap/read/refresh, WS echo/stream, bounded attachment envelopes,
   image/audio/video ranges and a harmless mutation counter; use synthetic identities/data and test-only secrets.
2. Implement loopback-only fixture startup with explicit bind/ports, private temp data, health and teardown;
   keep dependencies in the fixture's own project. It must not import or compose `AgentLoop`/gateway state.
3. Add deterministic modes for expired/revoked/wrong-instance credentials, lost/delayed responses,
   refresh rotation, WS disconnect, corrupt/oversized frames, redirects and cancel/backpressure.
4. Serve known-byte media with MIME, range/cancellation/error behavior and untrusted-content samples.
   Add TLS-negative fixture cases without teaching the release app to trust a debug CA or disable TLS.
5. Document runner inputs, fixture reset/effect-count observation and private-entry setup expectations;
   live HTTPS enrollment is separately opted in, not invented in code or committed to the fixture.

**Checks:** fixture checks fail against absent behavior, then pass; test range bounds/416, bad methods,
unknown routes, malformed JSON and replay counters. Verify no agent/provider imports or external calls.
**Gate/evidence:** fixture tests and deterministic state reset pass; retain contract hash, synthetic media
checksums, failing/passing logs, bind/cleanup evidence and an explicit “fixture, not backend” label.
**Rollback/inputs:** stop only the recorded fixture process and delete its temp data; local tests need no
tailnet credentials. Live endpoint/enrollment can remain pending while M03/M04 are implemented locally.

<a id="m03"></a>
## M03 — One embedded userspace transport and narrow bridge (P0)

**Objective/prerequisites:** [M02](#m02); prove the accepted Go/C ABI/Kotlin seam without a WebView VPN assumption.
**Allowed:** proposed `mobile/native-network/{go.mod,go.sum,*.go,include/}`, Android `app/src/main/cpp/`
and the generated package's `network/` Kotlin module, `mobile/src/native-broker.ts`, native tests/build scripts.
**Forbidden:** a second Tailscale node, raw URL/dial/LocalAPI APIs, general shell/filesystem, production WebUI.

1. Freeze a small typed C ABI for start/state/login, named HTTP operations, one WS, cancel and opaque handles;
   specify ownership/freeing, sizes, errors and event threading before writing JNI/plugin glue.
2. Implement one app-owned `tsnet.Server`; route HTTP TLS and WS through its dial path using standard hostname,
   SNI/certificate validation. Interactive public login opens only validated URLs in the system browser.
3. Implement Kotlin/JNI/Capacitor glue with bounded frames/queues and cancellation; JS receives state/data
   approved by schemas, never raw credentials, LocalAPI or a general network client.
4. Enforce exact confirmed origin/installation/port/method/path rules, bounded payloads and redirect checks
   at every hop. Deny arbitrary schemes/hosts/paths, localhost and cross-origin redirect navigation.
5. Add lifecycle shutdown/resume, WS backpressure and failed-node behavior. Any optional 10.0.2.2 fixture
   adapter is visibly debug-only and excluded from release; private calls never fall back to host transport.

**Checks:** Go/native unit tests and x86_64 Android instrumentation; red malformed-host/redirect/oversize,
TLS/SNI mismatch, queue-full and unavailable-node tests. Exercise memory ownership and duplicate start/stop.
**Gate/evidence:** one-node native HTTP/WSS/TLS works in local tests; denial matrix and ABI symbol/library
hashes retained. Live embedded path proof remains M05, not inferred from these tests.
**Rollback/inputs:** close node/sockets and unload only task resources. If core integration fails, record
evidence for accepted libtailscale/native HTTP-WS fallback; neither fallback success nor framework change is assumed.

<a id="m04"></a>
## M04 — Android Keystore and node-state custody (P0)

**Objective/prerequisites:** [M03](#m03); credentials and sensitive node state survive only the intended app lifecycle.
**Allowed:** proposed Android `custody/` Kotlin module/tests, `AndroidManifest.xml`, `res/xml/backup_rules.xml`,
`res/xml/data_extraction_rules.xml`, native core persistence adapter and bridge status types. **Forbidden:** JS secrets.

1. Generate an app-private Android Keystore encryption key and bounded versioned per-installation records;
   separate target credentials, installation binding and transport node state. Define unknown-version failure.
2. Persist refresh material before acknowledging pairing; serialize refresh writes and make crash/partial-write
   recovery fail closed. Use supported encrypted node-state persistence; document any measured library limitation.
3. Exclude credentials/node state/private previews from backup and restore; handle lock state, process death,
   upgrade and restored files without silently adopting a device identity or recreating valid credentials.
4. Implement native-only access/refresh use, reachable logout/revoke and always-local erasure; offline logout
   explicitly reports server revocation pending. Purge media handles/temp files and abort pending requests.
5. Sanitize bridge responses/logging/crash diagnostics and test app data/URL/clipboard surfaces for leaks;
   record emulator Keystore functionality separately from later hardware-backed custody evidence.

**Checks:** Android unit/instrumented death/restart/update/restore/reinstall and key-invalidated tests;
negative corrupt ciphertext/wrong installation/missing key must require fresh sign-in/pairing, not reset authority.
**Gate/evidence:** no reusable secret appears in JS/storage/logs/backups; state transitions and atomic recovery
pass with sanitized storage/backup inspection. Emulator API behavior is the only custody claim at this stage.
**Rollback/inputs:** revoke test identities when reachable, erase native app state and node enrollment using
only owned resources. Unavailable hardware custody is a later M21 input, not a P0 prerequisite.

<a id="m05"></a>
## M05 — Emulator transport/media/deep-link feasibility gate (P0)

**Objective/prerequisites:** [M02](#m02)–[M04](#m04) plus a recorded, locally approved private fixture endpoint/enrollment.
**Allowed:** proposed native `media/` and `links/` modules, fixture UI/tests and `mobile/scripts/qualify-emulator`;
P0 still owns only `mobile/`. **Forbidden:** production gateway/UI changes, global VPN disable or silent ACL widening.

1. Record the fixture origin, owned HTTPS mapping/policy diff and recovery; obtain local opt-in before live
   network changes. Keep standalone Tailscale absent/disabled in the guest without disabling the host's node.
2. Perform interactive embedded login; correlate app-node/peer identity with backend-observed path evidence
   or an approved isolated ACL fixture. A client assertion, 10.0.2.2 or inherited host routes are insufficient.
3. Exercise native bootstrap/refresh/read/WSS, normal TLS rejection, bounded attachments and image/audio/video
   ranges/cancel. Use app-origin resource handles or bounded native previews; keep bearers out of DOM URLs.
4. Parse versioned pairing/chat deep links; reject arbitrary URLs, origins/installations and navigation.
   Test untrusted HTML/SVG/download isolation, queue pressure, suspend/resume, simulated network loss and death.
5. Disable the embedded node and prove private requests fail despite host Tailscale. Rebuild x86_64 and
   arm64 artifacts, scan release inputs for debug route/trust leakage, and execute complete cleanup.

**Checks:** proposed fixture-phase emulator runner plus native/fixture tests; negative route/redirect/TLS,
revoked media/ranges, backup/death and transport-off cases must fail for their intended reason.
**Gate/evidence:** all accepted P0 criteria pass with app-node + backend path proof, screenshots/traces,
API/WebView/ABI/acceleration facts, arm64 build-only label and mapping/enrollment cleanup. Otherwise no broad integration.
**Rollback/inputs:** revoke test enrollment, close handles and restore only owned mappings. If live setup is
missing, retain PENDING while local tests continue. If both accepted transport approaches fail, return measured options.

<a id="m06"></a>
## M06 — Versioned host-neutral protocol and profile contracts (P1)

**Objective/prerequisites:** [M05](#m05); freeze inputs for independent backend/native owners while staging evolves.
**Allowed:** proposed `nanobot/remote/{__init__,protocol}.py`, `tests/remote/test_protocol.py`,
`mobile/contracts/mobile-v1.json` and contract fixtures. **Forbidden:** permission/UI rewrites, live host provisioning.

1. Enumerate method/path/action/schema/scope/limit/error rules under `/mobile/v1/*` and `/remote/v1/*`;
   map only required chat/history/media, pairing/refresh/self-logout and status operations. Unknown actions deny.
2. Specify `mobile_protocol=1`, supported range/bridge version, mandatory chat capabilities, installation ID,
   gateway generation, principal/scopes/expiry, server WS epoch, frame/attachment/media limits and optional lifecycle.
3. Define immutable `(installation_id, gateway_generation, request_id)` plus stable principal/action/digest;
   distinguish `submit` and `replay_only`, current-epoch admission and `409 outcome_unknown` without execution.
4. Define per-target private-origin/profile/custody binding, one active connection, explicit switch cleanup,
   identity-mismatch/update-required states and separate gateway-online/model-unavailable semantics.
5. Add cross-language contract vectors: a chat-compatible data-plane backend with lifecycle unsupported
   must work; incompatible mandatory chat must deny sends. Lock proposed policy defaults in tests.

**Checks:** backend contract tests and schema/vector checks; negative origin/installation/profile mixing,
missing mandatory fields, unsupported versions and falsely claimed local/native capabilities deny.
**Gate/evidence:** versioned route/scope/schema registry and fixtures pass; access versions are never operation
identity. Defaults are 120-second challenge, access at most 10 minutes, refresh expiry after 30 days inactivity.
**Rollback/inputs:** contract changes require coordinated native/backend fixtures, not implicit compatibility.
No live VPS/model or current staging helper names are needed; freeze first production full-source type evidence here.

<a id="m07"></a>
## M07 — Independent RemoteAccessOwner store and IPC (P1)

**Objective/prerequisites:** [M06](#m06); durable device authority exists independently of the gateway/agent.
**Allowed:** proposed `nanobot/remote/{store,owner,ipc,client}.py`, `tests/remote/test_{store,owner_ipc}.py`,
minimal explicit entries in existing `nanobot/config/schema.py`. **Forbidden:** agent/session copies or PID ownership DBs.

1. Resolve `<selected-instance-root>/remote-access/` through existing config/instance path helpers; default off.
   Persist versioned stable installation/device/revision/scope/status/credential-hash and owned-entry metadata.
2. Enforce directory 0700/files 0600, bounded parsing, high-entropy secrets, locks and atomic fsync/replace;
   corruption/unknown versions fail closed and never silently erase revocations or regenerate installation identity.
3. Implement a separate owner process and private Unix-socket typed IPC with selected-OS-user peer checks;
   distinguish gateway validation/revoke subscriptions from admitted local administrative methods.
4. Implement validation and revocation event contracts, restart recovery and bounded liveness detection;
   gateway loss of owner must deny mobile operations/close sessions while local clients remain available.
5. Add concurrent writers/crash-boundary/permissions/corrupt-record/unknown-version and unauthorized-peer
   tests; exercise owner restart with gateway absent and guarantee no import starts an agent runtime.

**Checks:** backend store/IPC tests plus focused config tests and full-source type comparison. Negative:
wrong OS user/role, admin methods through gateway IPC, symlink/path escape and unavailable owner all deny.
**Gate/evidence:** durable installation/revocations and role-bound IPC pass; record schema hashes, file modes,
fault-injection results and owner-only startup/cleanup evidence. No live service installation is needed.
**Rollback/inputs:** disable mobile ingress before reverting incompatible owner code; retain device/audit/revoke
records and never downgrade them silently. Existing permission/WebUI work can continue independently until M10.

<a id="m08"></a>
## M08 — Pairing, rotating credentials and live revocation (P1)

**Objective/prerequisites:** [M07](#m07); device identity cannot be collected by a second QR scanner or resurrected.
**Allowed:** proposed `nanobot/remote/{pairing,auth,devices}.py`, owner/IPC extensions and
`tests/remote/test_{pairing,auth,revocation}.py`. **Forbidden:** DM `pairing/store.py` reuse or legacy broad-token exchange.

1. Create local-operator challenges containing protocol/exact HTTPS origin/installation/random challenge/expiry;
   rate-limit attempts and bind each phone request to nonce, display label and native-only collector secret.
2. Show comparison code and explicit shared history/media/chat consent; approve the exact pending request
   through admitted local management. Atomically allow one challenge winner and one correct collector only.
3. Issue hashed opaque per-device refresh/access material with M06 lifetimes/scopes; serialize atomic rotation.
   A lost rotation response may require fresh pairing; never cache reusable bearer responses or retry blindly.
4. Enforce current status/scopes before introspection and revoke: deny new admissions, retire registered mobile
   sessions, then acknowledge. Preserve already admitted server work; self-logout grants no device administration.
5. Test access renewal, refresh expiry/uncertainty, scope reduction, owner restart, OS restore/reinstall and
   concurrent QR consume/collect/refresh/revoke races with deterministic clocks and barriers.

**Checks:** backend tests; wrong collector/nonce, replayed/expired QR, mismatched instance, revoked/expired
refresh, corrupted records and scope escalation issue nothing. Revoke/IPC loss closure target is at most 2 seconds.
**Gate/evidence:** one concurrent pairing winner, no stolen collection or secret cache, atomic rotation and
revocation linearization pass. Retain race traces and timing/counter evidence without QR/credential material.
**Rollback/inputs:** disable ingress, revoke test devices and retain revocation records. Comparison-code UI can
remain CLI-only until M18; product settings or service supervision are not prerequisites.

<a id="m09"></a>
## M09 — Staged owner listener, approval CLI and owned private entry (P1)

**Objective/prerequisites:** [M08](#m08); make the real owner reachable for P3 without P4/P5a or production installation.
**Allowed:** proposed `nanobot/remote/{http,staging,entry}.py`, owner/IPC client extensions,
`tests/remote/test_{listener,staging,entry}.py` and staging runbook. **Forbidden:** new agent instance or product service install.

1. Add a minimal loopback-only `/remote/v1/*` listener using existing transport dependencies/conventions;
   distinguish public pairing challenge possession from authenticated device and local-admin operations.
2. Implement proposed `python -m nanobot.remote.staging` help/dry-run/start/status/approve/teardown with explicit
   config/instance/listener/port selectors, private temp runtime paths and finite independently owned lifetime.
3. Validate TLS/HTTPS/tailnet readiness and installed Serve CLI behavior; reject trusted-proxy conflicts,
   conflicting mappings, nonloopback/public bind, Funnel and insecure overrides. Record prior and desired exact entries.
4. After concrete local opt-in, apply only owned prefix mappings and verify the real owner/control prefix
   remains ready with the gateway stopped. Approve an emulator request locally without desktop UI/production config.
   Data-prefix/WS plumbing may be probed against M02's non-agent echo fixture only, labeled fixture evidence;
   actual mobile gateway data/WS/media forwarding is verified in M11 after its routes exist, before M14.
5. Reconcile off/teardown with observed state, preserve other mappings and report unconfirmed acknowledgments;
   rehearse startup failure, partial setup and idempotent cleanup. Never use global `serve reset` or stop Tailscale.

**Checks:** backend listener/CLI tests with command adapter fakes, then opted-in real HTTPS/mapping probes;
spoofed Host/forwarded/Tailscale identity headers must not create local/device authority.
**Gate/evidence:** independent real owner reachability and exact restoration pass before M14; retain sanitized
prior/desired/final mapping diffs, selectors, readiness/teardown commands and approval results, not credentials.
**Rollback/inputs:** missing approved origin/enrollment/policy remains PENDING; local implementation/tests can
finish and M10 proceed. Only the operator's concrete opted-in resources may be changed; owner code stays with P1 through M15.

<a id="m10"></a>
## M10 — Clean committed staging-sync checkpoint

**Objective/prerequisites:** [M01](#m01) and committed permission/WebUI fixes with their affected tests passing.
**Allowed:** serialized integration of the mobile branch and its manifests/docs only; a deliberate rebase/update
onto the recorded committed staging SHA. **Forbidden:** editing/staging/resetting the staging checkout or copying dirty files.

The clean 2026-10-31 rebase already includes staging `a60d8c5037479848d985475eb2e1bc81159d1bb7`.
That completes the Git-sync portion only. At M10 verify ancestry and perform the remaining source,
permission/API/layout, test and type-evidence checks; no mobile implementation was tested by the rebase.

1. Check that all mobile writers have reached a clean committed checkpoint; record actual branch/HEAD/status,
   committed staging target and passing upstream affected-test evidence. Do not manufacture missing commits.
2. Freeze pre-sync source/dependency/type/contract manifests and compare committed permission/API/runtime/layout
   seams. Active staging files and its running server are not authoritative source/test fixtures for this task.
3. If the selected staging target is already included, record ancestry without repeating the rebase.
   Otherwise update only `codex/mobile-client`, resolving mobile conflicts in this checkout while preserving
   accepted security clauses and unrelated changes. Record old/new bases and the exact integration diff.
4. Re-read admitted permission owner/types, scoped API, runtime/bootstrap and layout contracts; revise mobile
   adapters/tests to current semantics through their owning tasks, not by forcing obsolete helper names/labels.
5. Retain all original pre-mobile and pre-sync checker reports/source snapshots and their dependency hashes.
   Record the new committed staging base/environment explicitly; compare that staging source without mobile
   changes to the rebased mobile source under identical new dependencies, retaining both full reports.
   Link old/new diagnostic signatures and rerun affected checks; never make a mobile candidate the new baseline
   or refresh a baseline to absorb mobile-introduced diagnostics, even if dependencies or staging changed.

**Checks:** clean status/ancestry/diff/manifests and relevant changed-seam tests; negative missing committed
input, dirty boundary or new diagnostic signature blocks completion. Dependency changes get a new explicit pair of snapshots.
**Gate/evidence:** clean sync and refreshed permission/API/layout assumptions are recorded before M11/M16.
Retain old/new comparison pairs, committed source evidence, conflict decisions and test exits;
unresolved mobile-introduced diagnostics stay failed across sync. No staging-runtime claim is inferred.
**Rollback/inputs:** preserve original committed mobile reference and follow normal integration rollback in
this checkout only. Awaiting committed fixes blocks shared integration; M02–M09 independent local work can continue.

<a id="m11"></a>
## M11 — Scoped gateway HTTP, WS and media routes (P2)

**Objective/prerequisites:** [M09](#m09) and [M10](#m10); authenticate the remote edge without broadening legacy/local entry.
**Allowed:** existing `nanobot/webui/{gateway_endpoint,gateway_tokens,ws_http,gateway_services,media_gateway,media_api,attachment_ingress}.py`,
`nanobot/channels/websocket/runtime.py`, remote client and matching WebUI/remote tests. **Forbidden:** runner/provider edits.

1. Wire typed owner IPC into gateway composition; register only M06 `/mobile/v1/*` operations and mandatory
   handshake. Keep local/browser/desktop paths and tokens intact; phone credentials never mint broad tokens.
2. Authenticate current principal/installation/scopes/expiry for every HTTP read, bootstrap, WS open/frame,
   media/range and attachment; bind server-issued socket epoch/gateway generation and enforce advertised limits.
3. Reject unknown routes/actions and nonlocal access to legacy bootstrap/settings/native endpoints, even through
   an entry mapping mistake or spoofed proxy headers. Refuse conflicting trusted-proxy mobile setup.
4. Enforce media authorization in addition to signed URL checks, including ranges, projection and cancellation;
   preserve attachment ingress envelope/limits. No new M1 upload endpoint or bearer DOM URL is needed.
5. Add expiry/revocation timers and owner-event retirement; loss of IPC fails closed within the tested bound.
   Verify local clients remain usable and persistent chat detach does not cancel server work.
6. With the isolated real gateway now running, verify actual owned data-prefix HTTP/WS/media forwarding and
   owner/control reachability. Distinguish this from M09's optional non-agent fixture probe before M14 admission.

**Checks:** backend checks plus existing HTTP-utils/media/attachment/replay-auth tests. Negative missing,
expired/revoked/wrong-instance/wrong-scope tokens must fail every allowed edge and forged local/proxy paths.
**Gate/evidence:** route-by-route/frame/range matrix passes with live owner denial/closure evidence; retain
schema/route hashes, scoped traces and full-source type comparison. Legacy signatures alone never authorize remote media.
**Rollback/inputs:** disable mobile admissions and owned ingress before incompatible rollback; keep local routes
and revocations. Backend ownership remains sequential through M13; a permission conflict goes to M13, not a core bypass.

<a id="m12"></a>
## M12 — Immutable operations and replay-only admission (P2)

**Objective/prerequisites:** [M11](#m11); reconnect after an uncertain send must never execute a second effect.
**Allowed:** existing `nanobot/webui/inbound_commands.py`, necessary M11 edge/token modules,
proposed remote replay helpers/tests and existing `tests/webui/test_webui_mutation_replay_auth.py`.
**Forbidden:** shared `nanobot-client.ts` writes (M16 owns them), cache keys based on credential versions or client IDs.

1. Bind cache records to immutable installation/generation/request plus stable principal/action/payload digest;
   check current authorization before lookup, join and result delivery, preserving existing result correlation.
2. Admit a new record only for `submit` at the current socket's server-issued epoch. `replay_only` can only
   join/return an existing record; cache miss/eviction is `409 outcome_unknown` with effect count zero.
3. Reject stale serialized submit epochs, changed principal/action/digest and gateway-generation mismatch before
   lookup/execution. No new request ID or epoch repair may retry an uncertain operation automatically.
4. Build harmless-counter tests losing responses at in-flight and completed stages, then separately renew access
   and rotate refresh for the same principal; preserve original task/result and effect count exactly one.
5. Race revoke/reduced scope with join/delivery and rotate/owner restart; deny latest unauthorized delivery with
   no new effect. Publish native/TS test vectors for M14/M16, including the existing `handleOpen` resend hazard.

**Checks:** deterministic barrier/clock replay tests plus existing mutation suite; negative eviction, old epoch,
restart, altered digest and different device fail without invocation. Measure admission/delivery ordering, not just HTTP status.
**Gate/evidence:** every lost-response/renewal/rotation combination has effect-count 1; unauthorized/replay-miss
cases create no execution. Retain counters, operation/principal correlation and frozen type/regression reports.
**Rollback/inputs:** disable mobile protocol before reverting replay semantics; retain durable device revocation.
No UI/backend network input blocks these tests. M16 must consume these vectors rather than inventing its own replay model.

<a id="m13"></a>
## M13 — Qualified remote task and media authority ceiling (P2)

**Objective/prerequisites:** [M12](#m12) and refreshed [M10](#m10) permission semantics; phone ingress must stay restricted.
**Allowed:** existing `nanobot/webui/{workspaces,session_projection,inbound_commands,media_gateway,ws_http}.py`,
channel edge and corresponding permission/native-admission tests. **Forbidden:** core runner/provider or unassigned permission-owner edits.

1. Trace current restricted/default managed-task admission from the updated permission owner; document exact
   source/task/revision binding. Use existing authority seams, not an obsolete helper name or broad boolean.
2. Intersect mobile requests with server-selected restricted authority at dispatch; deny Full/native references,
   arbitrary workspace changes, settings/system commands, temporary chat/delete/automation/skills/package management.
3. Continue existing chats only when their current task/scope qualifies; otherwise allow authorized reads and
   durable “Approval is needed on the server” guidance. Never silently rewrite focus/grants or open a folder picker.
4. Recheck authority at media projection/reconnect and concurrent local permission edits; persistent sessions
   cannot inherit a live desktop Full reference or global grant. Local/native flows retain their source binding.
5. Test unconfigured/unmanaged installations: advertise sends/target actions unsupported until a qualified local
   boundary exists. If edge seams cannot enforce this, stop that criterion for a bounded security revision.

**Checks:** current permission API/authority/native-admission tests plus remote ceiling cases; test malicious
scope/native fields, stale source/revision, concurrent desktop grants and unauthorized media/history projection.
**Gate/evidence:** no remote privilege inheritance/side-effect escape and supported restricted chat remains usable;
retain authority path diagram, negative matrix, local regressions and type evidence.
**Rollback/inputs:** disable mobile sends without weakening local permissions; preserve reads only if independently
authorized. Any required session/permission-owner file extension needs a named single-writer handoff before editing it.

<a id="m14"></a>
## M14 — Real-backend Android identity and reconnect qualification (P3)

**Objective/prerequisites:** [M05](#m05), [M09](#m09) live setup and [M13](#m13); turn fixture-qualified transport into the product client.
**Allowed:** `mobile/` native profile/pair/auth/connection modules, minimal product harness, contract vectors and
backend-phase emulator runner. **Forbidden:** shared WebUI, backend ownership or reuse of staging's installed data/environment.

1. Implement the M06 per-target profile in native custody: exact HTTPS origin, stable installation/capabilities,
   label and credentials; one active connection. A hostname identity reset requires explicit fresh pairing.
2. Pair through the real independent owner/local CLI, persist before acknowledgment and negotiate mandatory
   chat separately from optional lifecycle. Confirm compatible chat works when lifecycle is unsupported.
3. Exercise real access expiry/refresh/rotation/revoke/owner IPC loss and offline logout; uncertain refresh leads
   to re-pair guidance, not reusable old credentials. Close sockets/handles/pending operations on explicit target switch.
4. Drive M12 submit/replay vectors from the native harness against a harmless real-backend operation: lose
   responses in-flight/completed, renew/rotate, reconnect and verify effect-count 1 with stable identity.
5. Repeat app-node + backend path proof, TLS/origin mismatch, transport-off and process-death/reconnect tests;
   keep the artifact debug/internal and preserve persistent server work while the phone detaches.

**Checks:** proposed backend-phase emulator runner plus native/backend tests; negative cross-origin/profile
credential reuse, installation mismatch, old generation, revoked/reduced-scope delivery and silent failover all deny.
**Gate/evidence:** real-backend pairing/identity/replay matrix passes in x86_64 emulator with independent owner
and native-path traces. No physical device or live VPS/model proof is implied.
**Rollback/inputs:** revoke enrolled test device, remove only opted-in mappings/temp instance and erase native
secrets. Missing real setup leaves P3 pending; native unit work can continue, but M16 cannot skip this integration gate.

<a id="m15"></a>
## M15 — Native product media, attachments and deep links (P3)

**Objective/prerequisites:** [M14](#m14); qualify private media/attachment UX through the real broker before shared UI wiring.
**Allowed:** Android `media/`, `links/`, picker/activity/permission modules, `mobile/src/native-broker.ts`,
native tests/runner and asset trust config. **Forbidden:** new general file/URL bridge or production upload API.

1. Replace fixture handles with session/installation-bound broker resources authorized per fetch/range;
   support MIME/seek/cancel and normal TLS. Choose bounded native preview if WebView resource playback fails.
2. Integrate one camera/library attachment trigger with explicit sources and native handles; preserve existing
   attachment envelope, file/count/frame limits and backpressure without unbounded base64 bridge copies.
3. Keep required temporary bytes app-private/backup-excluded/bounded/expiring; purge at logout, revoke,
   instance switch and startup. Do not persist transcripts/private preview caches or queue offline mutations.
4. Route versioned pairing/chat deep links through reconnect/authorization and exact target binding;
   block arbitrary navigation and privilege-bearing remote HTML/SVG. External links use the system browser.
5. Exercise interrupted native activities, permission denial/cancel, process death, offline/revoked ranges and
   target switch while uploading/playing; preserve chat identity and close all pending handles.

**Checks:** native tests plus real-backend image/audio/video range/seek/playback, oversize/cancel and hostile
content cases in emulator; inspect DOM/resource URLs, storage and logs for credentials/private residue.
**Gate/evidence:** product media/attachment/deep-link matrix passes with bounded memory/queues/cleanup and
documented native-preview fallback where needed. P1 transfers listener/entry ownership only after this gate.
**Rollback/inputs:** erase temporary media/revoke test devices and restore test entry. Real camera/library/OEM
behavior remains M21; missing emulator media support must be resolved or explicitly fail its required preview criterion.

<a id="m16"></a>
## M16 — Shared React runtime, HTTP/bootstrap and replay integration (P4)

**Objective/prerequisites:** [M10](#m10) and [M15](#m15); reuse the existing chat client with explicit mobile transport semantics.
**Allowed:** existing `webui/src/lib/{runtime,types,http,bootstrap,api,nanobot-client}.ts`, `App.tsx`, `main.tsx`,
`webui/vite.config.ts`, necessary `webui/package.json`/lock changes, proposed mobile adapter/tests and `mobile/` build wiring.
**Forbidden:** backend/permission edits, remote HTML loading, desktop host functions inferred from “native”.

1. Add an explicit mobile runtime surface/transport contract with named HTTP/socket/media operations and
   capability mapping; desktop/browser retain current behavior. Native mobile is never desktop permission authority.
2. Route mobile bootstrap/reads/refresh/socket operations through the broker; avoid browser fetch/private DNS,
   bootstrap-secret localStorage/query paths and bearer exposure. Reject unsupported named operations natively.
3. Extend existing `NanobotClient` pending records for mobile immutable identity/digest/attempted state;
   `handleOpen` converts attempted sends to `replay_only`, preserving request/generation and never replaying serialized submit.
4. Authenticate/reconcile canonical chat/turn state after reconnect/remount; outcome-unknown requests hydrate
   current authorized state and require explicit new actions. Bound queues and preserve persistent detach semantics.
5. Build packaged shared assets to `mobile/www`; disable SW registration/caches only on mobile and enforce
   packaged-asset CSP/navigation. Preserve browser PWA and normal `nanobot/web/dist` wheel builds.

**Checks:** WebUI focused bootstrap/runtime/client/SW/main-registration tests, new mobile vectors and build/lint;
negative DOM/storage credentials, raw fetch fallback, stale-epoch resend, generation reset and false native authority fail.
**Gate/evidence:** same shared client serves Android emulator with M12 counters; browser/desktop/PWA tests pass.
Retain build-output inventories, release navigation/SW scan, correlated replay traces and exact changed paths.
**Rollback/inputs:** restore mobile adapter/build wiring without replacing installed/browser bundles; clear mobile
SW remnants if any. M10 API/layout drift requires a fresh seam manifest, not edits in the staging checkout.

<a id="m17"></a>
## M17 — Shared mobile chat layout and connection states (P4)

**Objective/prerequisites:** [M16](#m16); deliver usable persistent chat without exposing unsupported management controls.
**Allowed:** `webui/src/App.tsx`, `globals.css`, `components/thread/`, `Sidebar.tsx`, `AttachmentTile.tsx`,
`MessageBubble.tsx`, required media/rendering components, UI state/tests and mobile bridge presentation only.
**Forbidden:** backend grant/schema changes, temporary chats or a second mobile chat/state implementation.

1. Adapt existing chat-first shell/sidebar/composer for safe areas, small screens, keyboard resizing, portrait/
   landscape and accessible status. Android Back closes top viewer/sheet before navigating.
2. Present sign-in, pending pairing/comparison/expiry, connecting, refresh uncertainty, revoke, protocol/instance
   mismatch, reconnect/outcome-unknown, entry-off, gateway offline/stopped and model failure distinctly.
3. Bind native previews/attachments/deep links to current chat; keep one source chooser, correct cancel/denial
   behavior and bounded loading/error states. Hide/deny unsupported lifecycle, folder picker and local-only controls.
4. Keep theme/font/navigation preferences on-device; permitted titles/model/sidebar mutations remain shared
   server state. Existing ineligible chats show durable local-approval guidance without rewriting their authority.
5. Reconnect/resume/remount during text/tool streaming and confirm active chat/turn IDs and canonical transcript
   converge; detach a phone while server work continues and rejoin without duplicated sends.

**Checks:** component/recovery tests, full WebUI checks and emulator screenshots/input traces; negative narrow
viewport/large font/keyboard/Back/denied capabilities and history-media consent states must remain usable.
**Gate/evidence:** accepted chat/media/navigation states pass with browser/desktop regressions; retain sanitized
screen matrix, accessibility checks, transcript/turn reconciliation and no-offline-cache inspection.
**Rollback/inputs:** revert owned UI changes and purge mobile transient resources only. No physical phone,
iOS, lifecycle adapter or new provider configuration is required for this emulator gate.

<a id="m18"></a>
## M18 — Local settings, device list, pairing/revoke and remote on/off (P5a)

**Objective/prerequisites:** [M17](#m17) plus recorded P1/P2 and P4 named-file ownership transfer.
**Allowed:** transferred `nanobot/remote/` entry/approval/management modules and config entries;
`nanobot/webui/settings_{routes,contracts,services,system}.py`, shared settings controller/contracts/page and
proposed `RemoteAccessSettings.tsx`/tests; minimal product local CLI registration. **Forbidden:** remote admin grants.

1. Record exact transferred files/current hashes and stop prior writers; reuse P1 setup/approval/entry code
   rather than implementing a second identity store or mapping controller. Keep module CLI usable headlessly.
2. Add strictly local authenticated device list, QR/compare/expiry/history-media consent, approve/cancel/revoke
   and sanitized audit/status. Remote identity/proxy headers cannot open enrollment/admin/folder management.
3. Add explicit selected-instance/private-entry setup showing exact mapping changes/recovery and initial off;
   enable only after local opt-in and current listener/permission/proxy readiness checks.
4. Implement off ordering: deny admissions/retire sockets, remove only owned **data and control** mappings,
   verify observed state and retain “unconfirmed” on lost acknowledgment. Re-enable requires local management.
5. Test revoke during chat/media/replay and owner/gateway unavailability; retain sessions/revocations and distinguish
   disconnect-phone, entry-off and gateway-stop. M1 advertises status only, with no runtime start/stop grant.

**Checks:** local settings/remote-entry tests plus WebUI and existing settings-route tests; negative remote,
CSRF/origin/proxy spoof, concurrent toggle/revoke, mapping conflict and lost-ack cases must deny or report honestly.
**Gate/evidence:** local pairing/revoke/on-off/recovery and headless CLI parity pass; retain transfer ledger,
local-vs-remote route matrix and prior/desired/observed mapping/cleanup evidence.
**Rollback/inputs:** default off/remove owned entry, preserve device/audit/revoke records and other Serve config.
Live mapping changes need concrete local opt-in; code/fixture tests continue without another per-task approval.

<a id="m19"></a>
## M19 — Opt-in independent controller supervision (P5a)

**Objective/prerequisites:** [M18](#m18); supervised device authority remains distinct from gateway lifetime.
**Allowed:** proposed `nanobot/remote/{supervision,cli}.py`, transferred entry/settings code, minimal existing
`nanobot/desktop/adapter.py` glue and matching remote/desktop tests. **Forbidden:** gateway service/PID adoption or lifecycle scope.

1. Define the exact selected-instance owner command/config/socket/listener and bounded user-service adapter;
   retain foreground staging mode. Generate a concrete unit/install/remove/status diff and rollback before live setup.
2. After local opt-in, install/start only the recorded controller resource; verify ownership/conflicts and that no
   second supervisor or agent is created. No arbitrary unit/command/PID selector is accepted from the phone.
3. Test owner restart/reboot/logout behavior under the actual user-service configuration; advertise measured
   availability honestly. Do not assume linger/pre-login availability or grant system privileges implicitly.
4. Stop/restart the gateway through existing local semantics and verify controller/status remains reachable;
   preserve desktop guarded Quit and ensure the controller does not auto-start the agent afterward.
5. Uninstall/disable only the opted-in controller and its owned entry; retain revocations, sessions and unrelated
   units/mappings. Reconcile partial install/start failure, stale unit state and lost setup acknowledgments.

**Checks:** service-adapter fakes and existing desktop/gateway lifecycle tests, then opted-in local independent
controller restart/cleanup rehearsal; wrong instance/unit/service-owned gateway must never be adopted.
**Gate/evidence:** independent supervision and exact cleanup pass with actual service/command/config hashes;
unqualified availability is PENDING, never an always-on guarantee. Lifecycle remains advertised unsupported.
**Rollback/inputs:** missing local service opt-in blocks only live criterion; code/tests and candidate evidence
can proceed. Unit installation, linger or privileges are genuine setup choices, not inferred from plan acceptance.

<a id="m20"></a>
## M20 — Automated regressions and integrated review-candidate evidence

**Objective/prerequisites:** [M19](#m19); prepare a reproducible integrated candidate for device tests and final review.
**Allowed:** test/evidence/runbook updates within handed-off ownership; product fixes return to their task owner.
**Forbidden:** new features, unrelated test/type-debt repair, merge/activation or an artificial human approval stop.

1. Freeze candidate source/head/dependencies and rerun route/auth/pairing/replay/task/media/owner/entry matrices;
   include all in-flight/completed renewal/rotation counters and expiry/revoke/IPC timing on the integrated app.
2. Run required full Python, WebUI test/lint/build, scoped native/fixture tests and full-source strict checks;
   retain identical-environment before/after regression comparison and distinguish baseline debt from introduced errors.
3. Re-run complete backend-phase emulator qualification, browser/desktop local flows, two-start/PID/service
   refusals and unsupported-lifecycle-compatible-chat fixtures. Ensure debug routes/SW/secret leaks are absent.
4. Prepare raw diff/call-path/threat-boundary evidence for independent owner/IPC/native transport/media/replay/
   task-ceiling review; list failures and external checks honestly. No independent human sign-off is required to run M21/M22.
5. Return defects to the named owner, repeat affected checks after fixes and freeze a new candidate manifest;
   keep sanitized screenshots/traces and install/revoke/entry teardown rehearsals linked to exact source/artifacts.

**Checks:** full gates plus complete candidate acceptance matrix; negative stale review/source hash, missing
full output, timeout/forced cleanup and mock-only live results cannot count as passing criteria.
**Gate/evidence:** required automated regressions and zero-new-signature checks pass with a complete candidate
bundle. Unresolved full-suite/strict failures remain release-blocking pending; no unrelated fixes or false “full green”.
**Rollback/inputs:** disable test ingress/revoke devices and restore owned state. This is evidence preparation,
not final acceptance: independent review gates merge/production activation after M22 and all subsequent fixes.

<a id="m21"></a>
## M21 — Physical Android pre-release checkpoint (P6a)

**Objective/prerequisites:** [M20](#m20) automated candidate, actual supported test phone and local test enrollment;
no human approval checkpoint is inserted before ordinary isolated device tests.
**Allowed:** proposed `mobile/scripts/qualify-device`, device tests/evidence/runbook; fixes return to native/backend/UI owners.
**Forbidden:** iPhone/Xcode prerequisite, claiming emulator/arm64 compile success as hardware qualification.

1. Record device/model/API/WebView/ABI/OS power settings, app/source hash and signing facts; execute arm64-v8a
   with standalone phone Tailscale disabled. Establish app-node/backend path evidence again on real networking.
2. Test cellular NAT versus Wi-Fi/handover, offline/sleep/Doze/OEM background/process recovery and long-running
   server chat detach/rejoin. No always-on socket/background execution promise is inferred from success.
3. Verify actual Keystore hardware security level and lock/key-invalidated behavior; test upgrade/reinstall/
   restore/revoke and fresh pairing, without assuming emulator key behavior proves hardware custody.
4. Exercise real camera/library permission/cancel/interrupted activity, all media ranges/seek/native previews,
   keyboard/rotation/Back/accessibility and external/deep links with exact chat/installation binding.
5. Run a recorded repeatable workload and inspect battery/thermal/background activity and bounded queues;
   runaway reconnect/work, sustained thermal warnings, leaked credentials or uncontrolled temp growth fail qualification.

**Checks:** proposed device runner plus signed install/update/native instrumented tests and sanitized manual
input/network traces. Test transport-off, revoked/reduced-scope ranges and uncertain sends under real handover.
**Gate/evidence:** all physical acceptance rows pass on the declared supported matrix, with hardware custody,
power/thermal measurements and arm64 execution evidence. Unsupported devices/absent phone remain PENDING.
**Rollback/inputs:** revoke test device/delete private previews, restore only owned test mappings and keep user
data safe. A changed APK or native fix requires affected device checks again before final M22 artifact acceptance.

<a id="m22"></a>
## M22 — Reproducible signed Android artifact and final review handoff (P6)

**Objective/prerequisites:** [M21](#m21) and selected Android signing/application identity; finish the Android path.
**Allowed:** `mobile/` release/build/notices/lock/evidence files and release runbook; no private signing key in source/artifacts.
**Forbidden:** store publication, merge/production activation, future-host work or provider/agent changes.

1. Freeze exact source/dependencies/toolchain, application ID/version/build inputs and external signing reference;
   build release from clean recorded inputs twice, explain signing/timestamp differences and compare normalized contents.
2. Audit dependency provenance/transitives/licenses/notices and package contents; exclude fixture/debug direct-host
   path, permissive TLS/network config, node/refresh state, private logs and mobile SW/private caches.
3. Sign using the selected custody process; verify APK signature/certificate fingerprint, install/update/reinstall,
   protocol compatibility/update-required denial and revoke. Repeat affected M21 cases on the exact release APK.
4. Rehearse application/backend rollback: disable ingress before an incompatible protocol downgrade, retain durable
   revocations/sessions/audit and remove only owned entry/controller resources. Never fall back to broad legacy tokens.
5. Assemble the backlog's full review handoff at the **actual final HEAD and APK hash after all fixes**;
   record pending external criteria and obtain independent integrated review before merge/production activation.

**Checks:** clean reproducibility/Gradle release/signature/APK SHA-256 plus final full gates and affected device
checks; altered source/artifact, missing signing input, unsupported protocol or missing cleanup evidence fails its row.
**Gate/evidence:** complete signed artifact/install/revoke/rollback evidence and final independent acceptance;
only then call the Android release candidate accepted. No task completion claim substitutes for raw review evidence.
**Rollback/inputs:** absent signing identity allows unsigned/debug build evidence only, with release PENDING.
Any post-review fix invalidates affected results/review and needs a refreshed final head/artifact manifest and review.

<a id="m23"></a>
## M23 — Guarded desktop gateway lifecycle (P5b, deferred)

**Objective/prerequisites:** separately scoped after [M19](#m19); local lifecycle grant before live controls,
with owner-specific review before activation/merge. Isolated implementation/tests need no per-task approval.
**Allowed future boundary:** remote lifecycle module, existing `nanobot/desktop/adapter.py`, `nanobot/gateway/runtime.py`,
matching lifecycle tests and UI capability controls after handoff. **Forbidden:** service/foreign/foreground PID adoption.

1. Freeze the selected desktop-owned instance/selector/command/port/generation and fixed start/status/stop
   contract; advertise lifecycle only when supported. Phone supplies no executable/config/path/PID selector.
2. Invoke existing `DesktopAdapter`/`GatewayRuntime` admission, handles and locks; retain local birth-identity,
   generation and ownership proof. Refuse unknown/foreign/conflicting/service-owned/foreground processes.
3. Add separately granted current device scope and owner-side authorization/replay checks for controls;
   controller stays reachable for Start/status when the admitted gateway stops.
4. Test two-start/races/stale handles/PID reuse/generation changes, revoke and lost responses; preserve desktop
   Quit semantics and never auto-start a gateway after Quit or start a competitor after refusal.
5. Expose capability-dependent UI only after lifecycle tests and independent review; compatible chat remains
   enabled on unsupported hosts. Exercise exact off/recovery separately from gateway stop.

**Checks:** existing desktop adapter/gateway runtime tests plus future remote lifecycle counter/race tests;
arbitrary PID/unit/path and unsupported capability controls must deny regardless of client UI.
**Gate/evidence:** owner-specific admission/race/recovery matrix passes and fresh review precedes activation;
retain local handle proofs only in private evidence. No Android release dependency is added.
**Rollback/inputs:** revoke lifecycle scope/hide capability and keep independent owner/chat; do not kill a foreign
process. Future local scope and exact owner selection are required; no work is dispatched by the MVP prompt.

<a id="m24"></a>
## M24 — Exact registered service-manager lifecycle (P5b later, deferred)

**Objective/prerequisites:** [M23](#m23) plus separate named-unit/platform adapter contract and local service registration.
**Allowed future boundary:** proposed `nanobot/remote/service_lifecycle.py`, minimal existing `nanobot/gateway/service.py`
integration and service/remote tests after handoff. **Forbidden:** desktop PID admission, arbitrary unit/systemctl API.

1. Register locally the exact gateway service manager/unit/config/instance and separate controller service;
   keep service lifecycle unsupported until this adapter is qualified. Never infer authority from a PID/label.
2. Implement fixed start/stop/status through that manager and reconcile desired/observed state with finite
   operations; no signals beneath `Restart=always`, duplicate supervisor or competing foreground gateway.
3. Bind scopes/replay/generation to the selected registration; reject remote unit names, command text,
   changed configs, unrelated units and stale registration/manager identities.
4. Test Stop across `Restart=always`, races/reboot/manager failure/lost ack and external state changes;
   prove gateway stays stopped while the independent controller remains reachable for authenticated recovery.
5. Review the adapter independently, then advertise capability only for the qualified owner/platform;
   update operator recovery/unregister instructions and rehearse removal without affecting other services.

**Checks:** future fake-manager deterministic tests and opted-in exact-unit live rehearsal; service resurrection,
wrong unit/PID, controller stop coupling and arbitrary command injection must fail qualification.
**Gate/evidence:** exact-manager desired/observed/race/recovery results pass with unit/registration hashes and
independent review. Data-plane-only hosts continue chat with lifecycle unsupported.
**Rollback/inputs:** revoke scope/unregister adapter and retain controller/data plane; restore only selected unit
state after opt-in. Host/manager/unit/privileges are future inputs, not invented Android setup requirements.

<a id="m25"></a>
## M25 — Exact-target remote folder approvals (P7, deferred)

**Objective/prerequisites:** [M18](#m18) plus fresh bounded permission design and explicit local remote-approver grant.
**Allowed future boundary:** proposed remote permission adapter and closest current permission-owner/receipt tests
under a named-file handoff. **Forbidden:** global Full, opaque local-reference serialization or broad permission rewrites.

1. Freeze exact target/session/task/source/revision/read-write extent and expiry using the current local
   permission owner; define local scope grant/revoke and one pending target proposal, not machine administration.
2. Keep all native authority/references server-local; remote device receives a bounded proposal identifier
   and sanitized description. Local owner mints a single-use receipt for the exact current binding.
3. Validate current device scope, unchanged source/target/revision and receipt before invoking the existing
   owner locally; consume atomically and prevent replay/approval at another task/session/device.
4. Test changed source/target, stale revision, expired/reused receipt, another device and scope revoke racing
   consume/delivery; assert no filesystem/task effect outside the granted target/read-write extent.
5. Add narrowly scoped UI/audit and independent authority review; keep Full and all unrelated management local,
   then rehearse scope/receipt revocation while preserving ordinary restricted chat.

**Checks:** future authority/receipt/replay tests plus current native/permission regressions and full-source
type comparison; denial tests count effects, not merely hidden buttons or status codes.
**Gate/evidence:** every exact-binding/single-use/revoke negative passes; retain sanitized proposal/receipt
correlation and local owner review. No opaque reference or native capability crosses the wire.
**Rollback/inputs:** revoke approver scope/invalidate receipts and remove capability; keep normal chat and
local grants intact. Fresh accepted permission contract is required later; it never blocks M20–M22.

<a id="m26"></a>
## M26 — iOS adapter, custody, media and device qualification (P8, deferred)

**Objective/prerequisites:** future iOS scope, [M06](#m06)/[M16](#m16) shared contracts and an explicit shared-core handoff.
**Allowed future boundary:** proposed `mobile/ios/`, iOS build/test scripts and named shared transport/contract
files after Android writer handoff. **Forbidden:** Android support-floor changes or an iOS dependency edge into Android release.

1. Pin actual compatible Xcode/iOS/SDK/ABI/signing inputs and audit embedded transport options against the
   accepted narrow broker; reuse one userspace node, not a second wrapper/standalone VPN assumption.
2. Implement native adapter/custody with per-target Keychain/node-state policies and measured reinstall/restore/
   lock behavior; preserve native-only credentials and exact origin/route/queue/navigation boundaries.
3. Adapt HTTP/WSS/media/ranges/native preview, attachments/camera/library and deep links through the broker;
   keep packaged shared React assets, explicit mobile capability semantics and no mobile service worker.
4. Run simulator build/contract/negative tests, then actual signed-device network/handover/suspension/death/
   battery/custody/media/update/revoke checks; label simulated versus real evidence separately.
5. Produce reproducibility/notices/signing/review/rollback evidence for the actual iOS artifact and support matrix;
   any contract changes require Android regression checks through its current owner.

**Checks:** future iOS unit/simulator/device commands must be recorded after the project exists; reuse protocol/
replay fixtures, transport-off/native-path proof and authorization negatives. Simulator results cannot prove device custody.
**Gate/evidence:** actual qualified iOS transport/custody/media/signing/device matrix and independent review pass
before iOS support is claimed. Unavailable Mac/iPhone/signing remains a future pending input only.
**Rollback/inputs:** revoke iOS enrollment/erase app state and restore owned mappings; preserve server history/
revocations. iOS absence/failure never invalidates the independent Android release path.

<a id="m27"></a>
## M27 — Headless VPS data plane and permission qualification (future, deferred)

**Objective/prerequisites:** [M20](#m20) contract/regression evidence and separately selected host/deployment scope.
**Allowed future boundary:** proposed `docs/mobile-vps.md`, headless remote/operator tests and named deployment/
config files admitted later. **Forbidden:** invented VPS/credentials, Tauri requirement or hidden Android deployment work.

1. Inventory the selected host/instance/state/provider/permission capabilities and pin dependencies; choose
   one server agent/runtime and independent owner. Default to a new installation requiring fresh pairing.
2. Configure loopback gateway/owner and locally authenticated CLI/SSH pairing/revoke/on-off; apply only explicit
   private HTTPS mappings/policy after operator opt-in. No public listener/Funnel or remote unit/path controls.
3. Qualify the actual server-local task permission boundary; advertise sends/targets unsupported if not enforceable.
   Do not transfer desktop-native grants or substitute broad Linux filesystem access for an authority model.
4. Test mobile protocol/profile/identity, mandatory chat and status with lifecycle explicitly unsupported;
   separately prove existing gateway-online/model-unavailable states and server-only tools/files/history locality.
5. Qualify reboot/logout/owner recovery, revoke/entry-off and deployment rollback; add optional service lifecycle
   only by separately completing M24 for the exact registered manager/unit. No migration is implied.

**Checks:** future headless integration fixtures plus opted-in live host auth/permission/network/recovery tests;
unsupported lifecycle must hide/deny controls while chat succeeds, and unqualified task sends must deny.
**Gate/evidence:** selected headless deployment/permission/operator matrix passes with actual host/config hashes,
private path proof and cleanup. Fixture success alone is not a VPS deployment PASS.
**Rollback/inputs:** remove only opted-in deployment/entry/controller, retain backups/revocations and explicit
client profiles. Host, SSH/operator access, policy and topology are future inputs; none is requested for Android.

<a id="m28"></a>
## M28 — Private server-to-model data link (future, deferred)

**Objective/prerequisites:** [M20](#m20) and future model-host/topology scope; [M27](#m27) only if the selected server is a VPS.
**Allowed future boundary:** selected server provider config, proposed model-link runbook and isolated contract/tests;
existing provider factory/schema are read-only unless separately scoped. **Forbidden:** phone provider code/admin APIs/SSRF widening.

1. Select the actual inference endpoint/API/model/auth/timeout/header inputs on the server; confirm whether
   `auto`, `chat_completions` or `responses` matches that authenticated **data** API. Never invent credentials/ports.
2. Establish an intentionally served private model listener/proxy and least-privilege server-to-model policy;
   normal TLS/auth remain required. Tailnet connectivity alone does not expose model-host loopback.
3. Configure only server-side `api_base`/provider credentials; keep model load/unload/admin endpoints unreachable.
   Do not add broad CGNAT/web-tool whitelist exceptions, mount home files or grant home shell/browser execution.
4. Test inference/streaming/latency/timeouts/model-host sleep/auth failure while gateway stays connected;
   display inference unavailable separately from gateway offline. Cloud fallback requires explicit config/consent.
5. Rehearse removing the model link/restoring prior provider configuration and revoking its narrow credentials/
   policy, without affecting mobile identity/server sessions or other tailnet services.

**Checks:** isolated provider-topology fixtures, existing security URL-guard tests and opted-in authenticated
model data calls; wrong auth, admin access and tool SSRF to private ranges must remain denied.
**Gate/evidence:** actual authenticated model data/stream/failure distinction and unchanged web-tool security
pass with sanitized config/protocol hashes; no availability promise or private-message fallback is inferred.
**Rollback/inputs:** restore prior server config/remove owned mapping/policy only; keep keys out of logs/phone.
Actual server/model endpoints, operator policy and credentials are separately supplied future inputs.

<a id="m29"></a>
## M29 — Controlled state/identity migration and cutover (future, deferred)

**Objective/prerequisites:** [M22](#m22) evidence, [M27](#m27) if targeting VPS, and a separately approved exact-source/target migration scope.
**Allowed future boundary:** migration runbook/checks and explicitly named instance/state paths admitted later.
**Forbidden:** copying live competing writers, automatic profile failover/merge or native permission-authority transfer.

1. Inventory source/target data/config/session/skills/memory/device/audit/revoke versions and take verified
   backups. Choose explicit new-installation/fresh-pairing default or a separately qualified identity-continuation procedure.
2. Freeze ingress and quiesce the old authoritative writer; export/checksum/version-validate state using existing
   atomic persistence rules. Never serialize process-local native references or run two session/device authorities.
3. Import into isolated target paths with least privilege, preserve all revocations and audit, qualify task/model/
   owner/network behavior, and confirm durable installation/origin binding before enabling any client admission.
4. Cut over explicitly: stop old authority before target writes, enroll/update profiles with visible installation
   consent, test chat/history/media/replay and revoked devices. Never silently reuse credentials at another origin.
5. Rehearse rollback with one writer: disable target ingress, preserve any new writes/revocations and reconcile
   them under the agreed procedure before resuming source. A stale backup must never resurrect revoked credentials.

**Checks:** deterministic export/import/version/corruption/revoke/cutover tests and isolated rehearsal followed
by separately opted-in live checks; count writers and verify state hashes before/after every authority transition.
**Gate/evidence:** exact state/identity/pairing/cutover/rollback matrix passes with one authoritative writer,
restore-tested backups and operator-visible profile change; no silent identity adoption or grant transfer.
**Rollback/inputs:** restore only reconciled state with revoked identities still denied; retain both evidence sets.
Source/target, downtime/cutover choice, state paths and identity mode are future decisions, never Android prerequisites.

<a id="m30"></a>
## M30 — Owner scope on the existing pairing (owner-client revision)

**Objective/prerequisites:** accepted 2026-11-01 owner-client revision and [M14](#m14).
**Allowed boundary:** `nanobot/remote/{protocol,pairing,auth,store,management}.py`,
remote pairing tests, and the existing local Remote Access settings UI. **Forbidden:** a second
pairing flow or a separate admin credential.

1. Add the single `owner` scope to supported scopes and grant it with new pairings after the
   comparison-code approval. Keep chat/media scopes for compatibility.
2. Return `owner` in the handshake capabilities and expose it to the mobile runtime.
3. Keep legacy chat-only devices working; add an explicit re-pair/upgrade path rather than
   silently widening existing tokens.
4. State in the pairing UI that the phone receives full Nanobot control.
5. Test scope validation, approval, refresh/rotation, revoke, and legacy compatibility.

**Checks:** owner scope reaches the phone; non-owner/legacy tokens stay chat-only; revoke denies
owner and chat; pairing tests and token tests pass.
**Gate/evidence:** a real phone pairs once, handshake reports owner, and the same token is accepted
by the owner bridge; legacy denial tests remain green.
**Rollback/inputs:** revoke owner devices and re-pair; legacy devices are unaffected. Pairing still
requires local approval.

<a id="m31"></a>
## M31 — Full mobile WebUI read and mutation bridge

**Objective/prerequisites:** [M30](#m30).
**Allowed boundary:** `nanobot/remote/edge.py`, `composition.py`, `nanobot/webui/ws_http.py`,
`inbound_commands.py`, mobile runtime HTTP/WS tests. **Forbidden:** a second session/agent owner,
raw localhost proxy, or credential export.

1. Use the standard WebUI feature API. `/mobile/v1` remains the thin device-auth edge for
   pairing/refresh/handshake; mobile owner requests use the existing `/api/*` HTTP routes and
   the standard WebSocket path. Do not add a second feature-route namespace. Keep media binary
   routing in the same standard media route.
2. Validate the owner device token at the standard HTTP/WS boundary and propagate the owner
   identity so the existing `webui_request` mutation surface, settings handlers and local-only
   checks apply the owner policy. Desktop browser/token behavior remains unchanged.
3. Make workspace-control and Full-access authorization consult the owner scope instead of
   loopback-only checks; preserve source/revision/receipt validation.
4. Include private permission projections and session-update fields for owner mobile sockets.
5. Keep non-owner/legacy sockets denied; add positive and negative tests for every new route/action.

**Checks:** representative GETs, settings mutations, skill/automation actions, session delete,
workspace mode and Full access; unknown/oversized paths deny; legacy tokens remain denied.
**Gate/evidence:** full WebUI read/mutation matrix passes from the phone; provider keys never appear
in responses; owner and legacy authorization tests both pass.
**Rollback/inputs:** disable owner scope admission and revoke owner devices; desktop/browser flows
remain unchanged.

<a id="m32"></a>
## M32 — Full client UI, settings, workspace and Full access

**Objective/prerequisites:** [M31](#m31).
**Allowed boundary:** shared `webui/src` settings/sidebar/composer/workspace components and their
tests. **Forbidden:** a mobile-only fork of settings or hidden authority paths.

1. Re-enable Settings, Apps, Skills, Automations, Manage models and the full settings surface on
   mobile. Remove the model-only settings projection; use the full WebUI payload.
2. Use full workspace controls for owner phones. Keep the native folder picker unavailable to the
   phone and use the existing server-side path/management dialog instead.
3. Allow workspace mode and Full access from the phone with the existing confirmation dialogs.
4. Restore the full slash-command palette for owner clients, including restart/lifecycle commands
   exposed by the WebUI.
5. Keep the mobile composer layout responsive; settings pages must render at phone width.

**Checks:** settings pages load and save; folder path management works; Full access can be enabled
and returned; slash commands execute; phone-width UI tests pass.
**Gate/evidence:** real-phone settings/apps/skills/automations/full-access checks plus screenshots.
**Rollback/inputs:** hide the re-enabled surfaces and revoke owner access; server config remains
authoritative.

<a id="m33"></a>
## M33 — Owner-action safeguards

**Objective/prerequisites:** [M31](#m31).
**Allowed boundary:** shared confirmation UI, remote audit logging, owner mutation tests.
**Forbidden:** silent destructive mutations or provider-secret exposure.

1. Add explicit confirmation for session delete, skill/MCP/CLI install or removal, automation
   creation/run, Full access, remote-entry disable and runtime restart.
2. Audit owner mutations with installation ID, action, target and result; never log tokens or keys.
3. Keep provider API keys write-only; redact secrets from settings reads and errors.
4. Keep self-revoke available and ensure revoke invalidates owner access immediately.
5. Preserve short access-token TTL and refresh rotation; add optional step-up auth as a follow-up.

**Checks:** confirmation required; audit entry written; keys absent from all mobile responses;
self-revoke and remote-entry-off recovery guidance pass.
**Gate/evidence:** audit/redaction/confirmation tests and a live device revoke check.
**Rollback/inputs:** disable owner mutations and revoke owner devices.

<a id="m34"></a>
## M34 — Owner-client qualification and release

**Objective/prerequisites:** [M32](#m32), [M33](#m33).
**Allowed boundary:** tests/evidence/release artifacts.
**Forbidden:** claiming full parity without real-phone and full-suite evidence.

1. Run the full Python, WebUI, Go and Android suites; repair regressions through the owning tasks.
2. Re-pair the phone as an owner client and verify settings, skills, automations, workspace/Full,
   session delete, restart and remote-access self-revoke.
3. Verify update persistence (`adb install -r`) and one-time Tailscale re-auth behavior.
4. Build the signed release, run install/update/different-key rejection, and record evidence.
5. Update this taskbook, the mobile README and the final artifact hash; hand off for human acceptance.

**Checks:** full automated matrix, owner-client device matrix, update/re-auth evidence, signed artifact.
**Gate/evidence:** all gates green and final human acceptance. No owner-client parity claim before M34.
**Rollback/inputs:** restore the prior signed release and revoke owner devices; preserve server state.
