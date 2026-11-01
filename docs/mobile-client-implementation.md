# Mobile client: implementation steps for every task

This is the detailed companion to [the complete backlog](mobile-client-tasks.md).
Task IDs M01–M35 and anchors are stable; M36–M40 add missing bounded work. This revision ran only
documentation/static checks, not the product commands below. Proposed contracts/names must be verified
at M36: some implementation and evidence were reported historically, but were not inspected here. The [accepted plan](mobile-client-plan.md)
defines protocol/security behavior; this guide decomposes it without changing the architecture.
Implementation starts when the user invokes the backlog's OpenCode prompt; this planning task prepares it only.

## Common execution and evidence contract

M36 first inspects the actual implementation and freezes desktop parity and current source/evidence.
Prior implementation/test/APK claims are attributed historical reports, not accepted or rejected by this
revision. Retained M01–M22 work may be reused only after recorded applicability and affected requalification;
it is not an alternative chat-only release. Before each task, verify prerequisites/current source, record
owned paths and preserve others' edits. Capture a failing behavior test before implementation, implement minimally, refactor, then run
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

Run from this mobile worktree with its own environments. These commands are retained from repository
instructions and earlier inspected documentation, not freshly executed/verified here. M36/M10 must verify
current command/test names and record exact replacements if committed seams changed.

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

The following are **proposed interfaces**, whose existence/current behavior must be checked at M36. M01/M02 create the fixture/toolchain
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

- M36 is the early inventory/baseline owner: proposed `mobile/contracts/desktop-parity.json` and
  sanitized manifests only; no product edits. Its contract/evidence handoff precedes M01 and M06.
- M01–M05: one P0 writer under `mobile/` only. Review/requalify existing work rather than assuming absence.
  No production/root dependency files, installed app or staging runtime changes.
- M06–M13, M30/M31 and M37: one backend writer serially. M06 receives `mobile/contracts/` from P0;
  M31 additionally receives `mobile/contracts/operations-v1.json`, `mobile/src/native-broker.ts` and
  native registry files named in the M36 manifest. No overlapping native writer. Backend retains
  listener/approval/entry modules through M15. M10 is a serialized committed-source checkpoint.
- M14–M15: native product writer under `mobile/` after the registry handoff. M16–M17: shared WebUI/build
  writer, with explicit `mobile/` build-wiring handoff. Native mobile never implies desktop authority.
- M18–M19 receive exact remote/config/settings files from backend and shared UI writers. M32 receives
  shared UI only and consumes M37; M33 receives named confirmation/audit/permission/settings files.
  Record files/hashes and stop old writers before each transfer.
- M35 is isolated qualification; M20–M22/M34 verify the current integrated candidate. Product defects
  return to the owning task and invalidate affected evidence. No installed-instance mutation in these tasks.
- M38 is independent read-only review. M39/M40 alone own the reviewed installed transition/checks, after
  actual target selection and concrete install/config/service/network opt-in. These are not authorized now.
- M23–M29 remain separately scoped. They do not block existing supported owner actions or Android release.
  No concurrent changes to shared config, schema, registry, lockfiles or generated output.

The [taskbook](mobile-client-tasks.md#dependency-order-and-finish-lines) is the canonical DAG. The default
order is M36, M01–M13, M30/M31/M37, M14–M19, M32/M33/M35, M20–M22, M34/M38, then M39/M40
with their external gates. No early task requires a later phone bridge/physical qualification.

No task grants raw URL/dial/LocalAPI/general shell/filesystem access from the native bridge, privileged
remote HTML, offline private history/write queues, a second agent/runtime, exported provider/app
credentials, automatic cloud fallback, public ingress or home tools through a model connection.
Current owners **can** use temporary chats, settings, grants and Full through standard feature APIs and
server-local permission receipts; obsolete prohibitions apply only to the legacy profile/M25.
Newly entered provider keys may exist transiently in the settings form and standard write request;
never persist/log/replay/cache them or return existing secrets. Long-lived app credentials stay native.
No task authorizes merge/publication, installed activation without concrete opt-in, or invented
hosts/recipients/credentials. Framework/support-floor choices remain measured.

<a id="m36"></a>
## M36 — Freeze desktop parity/API and actual implementation baseline

**Objective/prerequisites:** none; perform the first fresh implementation inspection after this planning edit.
**Allowed:** proposed `mobile/contracts/desktop-parity.json`, sanitized `mobile/evidence/<run-id>/M36/`
and task applicability/evidence index; read source/tests/config schemas and historical reports only as
authorized implementation discovery. **Forbidden:** product edits, dependency installation, runtime/network/
installed-instance mutation, accepting historical tests/APKs by assertion or resetting checker baselines.

1. Record clean HEAD/tree, source/dependency/config/schema/tool/environment hashes, current desktop build and
   feature flags. Preserve unrelated edits. Inspect actual mobile/backend/shared implementation when work
   resumes; identify which proposed paths already exist and their exact current owners.
2. Enumerate every desktop feature/action, not only pages: chat/history/media/session/temporary behavior,
   settings/providers/models, Apps/MCP/CLI/API service, local/custom/installed skills and marketplace,
   automations, workspace/grants/modes/Full, device/entry management and existing supported lifecycle.
   Include supported imports/downloads/binary/external-auth flows. Derive exact routes/methods/WS actions.
3. For each inventory row record stable ID, feature/action, baseline/source owner, route+method or WS/control
   action, UI entry, owner/legacy auth, native operation/schema/limits, adaptation, dependency task, positive/
   negative checks, evidence/hash/status and explicit accepted exception. Record local-only initial setup,
   new-device comparison approval/scope-upgrade and unsupported remote Start as deliberate exceptions.
4. Map current source and historical M01–M22 evidence to each task/row: reusable only after applicability
   review and affected fresh checks. Record gaps without blanket absent-code or PASS assertions. Freeze exact
   file lists for later handler/UI/registry ownership; verify existing command/test names against instructions.
5. Freeze the inventory and source manifest for dispatch. Any new/changed operation updates the inventory
   and affected checks. An unsupported action needs explicit product acceptance before claiming full parity.

**Checks:** deterministic baseline manifests and exhaustive action→API→native→UI→evidence cross-check;
negative omitted category/action, missing method/scope/limit, unbound evidence or silent exception fails.
**Gate/evidence:** a versioned inventory with no unclassified action, exact source/baseline hashes, explicit
legacy/local-only exceptions and task applicability ledger. No implementation/test acceptance is inferred.
**Rollback/inputs:** restore only owned inventory changes if rejected, retain raw manifests/historical reports.
Actual implementation/desktop baseline access is required; no phone, pairing, enrollment or signing prerequisite.

<a id="m01"></a>
## M01 — Isolated Android shell, toolchain and evidence harness (P0)

**Objective/prerequisites:** [M36](#m36); establish reproducible inputs in the clean mobile worktree.
**Allowed:** proposed `mobile/{package.json,bun.lock,capacitor.config.ts,tsconfig.json,vite.config.ts,.gitignore,README.md}`,
`mobile/src/fixture/`, generated `mobile/android/`, `mobile/scripts/`, `mobile/toolchain/` and ignored evidence.
**Forbidden:** all production/root dependency files and the installed staging environment; retain P0-only ownership.

1. Run preflight; record actual clean HEAD, baseline ancestry, source hashes and M36's existing-project inventory.
   Reuse/requalify compatible scaffold/toolchain work; do not erase or duplicate existing implementation.
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

**Objective/prerequisites:** [M05](#m05) and [M36](#m36); freeze inputs for independent backend/native owners while staging evolves.
**Allowed:** proposed `nanobot/remote/{__init__,protocol}.py`, `tests/remote/test_protocol.py`,
`mobile/contracts/mobile-v1.json` and contract fixtures. **Forbidden:** permission/UI rewrites, live host provisioning.

1. Freeze the M36 inventory's standard `/api/*`, existing WS/media method/action/schema/scope/limit/error
   registry, plus independent `/remote/v1/*` auth/status/device-control operations. No duplicate feature API.
   Inventory negotiated legacy `/mobile/v1/*` aliases/shims and retain supported chat-only compatibility;
   removal needs a tested migration or explicit accepted update-required cutoff. Unknown operations deny.
2. Specify `mobile_protocol=1`, supported range/bridge version, mandatory owner/legacy feature capabilities, installation ID,
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
   distinguish gateway validation/revoke subscriptions, admitted local setup/enrollment and enumerated
   owner-delegated administrative envelopes. M30 supplies delegated operations, not a blanket IPC admin bypass.
4. Implement validation and revocation event contracts, restart recovery and bounded liveness detection;
   gateway loss of owner must deny mobile operations/close sessions while local clients remain available.
5. Add concurrent writers/crash-boundary/permissions/corrupt-record/unknown-version and unauthorized-peer
   tests; exercise owner restart with gateway absent and guarantee no import starts an agent runtime.

**Checks:** backend store/IPC tests plus focused config tests and full-source type comparison. Negative:
wrong OS user/role, unauthenticated/unlisted administration through gateway IPC, symlink/path escape
and unavailable owner deny. M30 must separately prove authenticated allowed owner delegation succeeds.
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
2. Show comparison code and explicit full-control/shared-history/media consent; approve the exact pending request
   through admitted local management. Atomically allow one challenge winner and one correct collector only.
3. Issue hashed opaque per-device refresh/access material with M06 lifetimes/scopes; serialize atomic rotation.
   A lost rotation response may require fresh pairing; never cache reusable bearer responses or retry blindly.
4. Enforce current status/scopes before introspection and revoke: deny new admissions, retire registered mobile
   sessions, then acknowledge. Preserve already admitted server work; self-logout alone grants no administration; M30 binds the single owner capability for new pairings.
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

**Objective/prerequisites:** [M01](#m01), [M36](#m36) and committed permission/WebUI fixes with their affected tests passing.
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

1. Wire typed owner IPC into gateway composition; register only M06 standard feature operations, independent control integration and negotiated legacy
   compatibility, with mandatory handshake. Keep local/browser/desktop paths and tokens intact; phone credentials never mint broad tokens.
2. Authenticate current principal/installation/scopes/expiry for every HTTP read, bootstrap, WS open/frame,
   media/range and attachment; bind server-issued socket epoch/gateway generation and enforce advertised limits.
3. Reject unknown routes/actions, broad legacy bootstrap exchange and unauthorized access to settings or
   local-only/native setup endpoints (M31 adds full owner feature authorization), even through
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

**Objective/prerequisites:** [M12](#m12) and refreshed [M10](#m10) permission semantics; owner and
legacy phone ingress must enforce the same current task/workspace policy.
**Allowed:** existing `nanobot/webui/{workspaces,session_projection,inbound_commands,media_gateway,ws_http}.py`,
channel edge and corresponding permission/native-admission tests. **Forbidden:** core runner/provider or unassigned permission-owner edits.

1. Trace current restricted/default managed-task admission from the updated permission owner; document exact
   source/task/revision binding. Use existing authority seams, not an obsolete helper name or broad boolean.
2. Intersect owner requests with current server-selected authority at dispatch. Owner clients may request
   Full/workspace/settings/temporary/delete/automation/skills/package actions; legacy chat-only clients deny
   those. Never serialize native references to either client.
3. Continue existing chats only when their current task/scope qualifies; otherwise allow authorized reads and
   durable “Approval is needed on the server” guidance. Never silently rewrite focus/grants or open a folder picker.
4. Recheck authority at media projection/reconnect and concurrent local permission edits; persistent sessions
   cannot inherit a live desktop Full reference or global grant. Local/native flows retain their source binding.
5. Test unconfigured/unmanaged installations: advertise sends/target actions unsupported until a qualified local
   boundary exists. If edge seams cannot enforce this, stop that criterion for a bounded security revision.

Owner cases here use the M06 typed identity/admission fixtures to prove the existing task boundary;
M30 supplies real owner delegation and M14 proves it end to end. This foundation gate does not require
later live owner pairing or falsely claim completed grant/Full delegation.

**Checks:** current permission API/authority/native-admission tests plus remote ceiling cases; test malicious
scope/native fields, stale source/revision, concurrent desktop grants and unauthorized media/history projection.
**Gate/evidence:** owner and legacy clients cannot escape the current policy/source binding; supported chat
remains usable for both;
retain authority path diagram, negative matrix, local regressions and type evidence.
**Rollback/inputs:** disable mobile sends without weakening local permissions; preserve reads only if independently
authorized. Any required session/permission-owner file extension needs a named single-writer handoff before editing it.

<a id="m14"></a>
## M14 — Real-backend Android identity and reconnect qualification (P3)

**Objective/prerequisites:** [M05](#m05), [M09](#m09), [M13](#m13) and [M31](#m31); qualify the
current owner backend in the emulator with the recorded live isolated setup.
**Allowed:** `mobile/` native profile/pair/auth/connection modules, minimal product harness, contract vectors and
backend-phase emulator runner. **Forbidden:** shared WebUI, backend ownership or reuse of staging's installed data/environment.

1. Implement the M06 per-target profile in native custody: exact HTTPS origin, stable installation/capabilities,
   label and credentials; one active connection. A hostname identity reset requires explicit fresh pairing.
2. Pair once as owner through the real independent owner/local comparison-code CLI, persist before
   acknowledgment and negotiate mandatory
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

**Objective/prerequisites:** [M16](#m16); adapt the shared chat shell and all connection states for current owner use.
**Allowed:** `webui/src/App.tsx`, `globals.css`, `components/thread/`, `Sidebar.tsx`, `AttachmentTile.tsx`,
`MessageBubble.tsx`, required media/rendering components, UI state/tests and mobile bridge presentation only.
**Forbidden:** backend grant/schema changes or a second mobile chat/state implementation. Temporary
owner chats retain standard connection-owned disposal; M32 covers the complete management-page adaptation.

1. Adapt existing chat-first shell/sidebar/composer for safe areas, small screens, keyboard resizing, portrait/
   landscape and accessible status. Android Back closes top viewer/sheet before navigating.
2. Present sign-in, pending pairing/comparison/expiry, connecting, refresh uncertainty, revoke, protocol/instance
   mismatch, reconnect/outcome-unknown, entry-off, gateway offline/stopped and model failure distinctly.
3. Bind native previews/attachments/deep links to current chat; keep one source chooser, correct cancel/denial
   behavior and bounded loading/error states. Owner clients use the full settings/lifecycle/workspace surface;
   the native server folder dialog remains unavailable to the phone and uses server-side path management.
4. Keep theme/font/navigation preferences on-device; permitted titles/model/sidebar mutations remain shared
   server state. Existing ineligible chats show durable local-approval guidance without rewriting their authority.
5. Reconnect/resume/remount during text/tool streaming and confirm active chat/turn IDs and canonical transcript
   converge; detach persistent chat while server work continues and rejoin without duplicated sends.
   Separately disconnect a temporary owner chat and prove disposal rather than persistent rehydration.

**Checks:** component/recovery tests, full WebUI checks and emulator screenshots/input traces; negative narrow
viewport/large font/keyboard/Back/denied capabilities and history-media consent states must remain usable.
**Gate/evidence:** accepted chat/media/navigation states pass with browser/desktop regressions; retain sanitized
screen matrix, accessibility checks, transcript/turn reconciliation and no-offline-cache inspection.
**Rollback/inputs:** revert owned UI changes and purge mobile transient resources only. No physical phone,
iOS, lifecycle adapter or new provider configuration is required for this emulator gate.

<a id="m18"></a>
## M18 — Local settings, device list, pairing/revoke and remote on/off (P5a)

**Objective/prerequisites:** [M17](#m17), [M31](#m31) plus recorded P1/P2 and P4 named-file ownership transfer.
**Allowed:** transferred `nanobot/remote/` entry/approval/management modules and config entries;
`nanobot/webui/settings_{routes,contracts,services,system}.py`, shared settings controller/contracts/page and
proposed `RemoteAccessSettings.tsx`/tests; minimal product local CLI registration. **Forbidden:** a second device
identity store or a second pairing flow; owner-client grants are handled by M30.

1. Record exact transferred files/current hashes and stop prior writers; reuse P1 setup/approval/entry code
   rather than implementing a second identity store or mapping controller. Keep module CLI usable headlessly.
2. Add authenticated device list, QR/compare/expiry/history-media/full-control consent, approve/cancel/revoke
   and sanitized audit/status. Initial setup, challenge creation/comparison approval and scope upgrade remain server-local;
   owner clients list/rename/revoke devices and manage already registered entry settings through M30 delegation.
   Pair-approval denial and owner-management success must be separate tested rows.
3. Add explicit selected-instance/private-entry setup showing exact mapping changes/recovery and initial off;
   enable only after local opt-in and current listener/permission/proxy readiness checks.
4. Implement off ordering: deny admissions/retire sockets, remove only owned **data and control** mappings,
   verify observed state and retain “unconfirmed” on lost acknowledgment. Re-enable requires local management.
5. Test revoke during chat/media/replay and owner/gateway unavailability; retain sessions/revocations and distinguish
   disconnect-phone, entry-off and gateway-stop. Owner clients use the WebUI lifecycle surface; service-manager
   lifecycle remains M23/M24.

**Checks:** local settings/remote-entry and WebUI/settings-route tests; positive current owner delegation,
negative unauthorized/legacy management, remote local-only enrollment, CSRF/origin/proxy spoof, concurrent toggle/revoke, mapping conflict and lost-ack cases must deny or report honestly.
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
   Record the selected gateway lifecycle owner and exact tested local launcher/recovery command for M32;
   unsupported remote Start stays explicit. M35 later verifies that command on the integrated candidate.
5. Uninstall/disable only the opted-in controller and its owned entry; retain revocations, sessions and unrelated
   units/mappings. Reconcile partial install/start failure, stale unit state and lost setup acknowledgments.

**Checks:** service-adapter fakes and existing desktop/gateway lifecycle tests, then opted-in local independent
controller restart/cleanup rehearsal; wrong instance/unit/service-owned gateway must never be adopted.
**Gate/evidence:** independent supervision and exact cleanup pass with actual service/command/config hashes;
unqualified availability is PENDING, never an always-on guarantee. Additional remote Start/service-manager capability remains unsupported; retain existing supported
WebUI restart/stop with their current process-ownership checks.
**Rollback/inputs:** missing local service opt-in blocks only live criterion; code/tests and candidate evidence
can proceed. Unit installation, linger or privileges are genuine setup choices, not inferred from plan acceptance.

<a id="m20"></a>
## M20 — Automated regressions and integrated review-candidate evidence

**Objective/prerequisites:** [M19](#m19) and [M35](#m35); qualify the current full-owner isolated
candidate before device/signing checks and later independent review.
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
not final acceptance: M34 completes parity/release evidence and M38 independently reviews it before
M39 installed promotion or any separately authorized merge/production activation.

<a id="m21"></a>
## M21 — Physical Android pre-release checkpoint (P6a)

**Objective/prerequisites:** [M20](#m20) automated candidate, actual supported test phone and local test enrollment;
no human approval checkpoint is inserted before ordinary isolated device tests.
**Allowed:** proposed `mobile/scripts/qualify-device`, device tests/evidence/runbook; fixes return to native/backend/UI owners.
**Forbidden:** iPhone/Xcode prerequisite, claiming emulator/arm64 compile success as hardware qualification.

1. Record device/model/API/WebView/ABI/OS power settings, app/source hash and signing facts; execute arm64-v8a
   with standalone phone Tailscale disabled. Establish app-node/backend path evidence again on real networking.
   Use the internally signed current candidate here; M22 supplies the final signing identity/artifact and
   repeats affected physical rows on that exact release. This gate does not require a later release APK.
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

**Objective/prerequisites:** [M21](#m21) and selected Android signing/application identity; package the
current owner candidate. This retained obligation is not the current product finish.
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
   record pending external criteria for M34 parity closure and M38 independent review before M39 promotion
   or separately authorized merge/production activation. Do not stop at the historical M22 finish.

**Checks:** clean reproducibility/Gradle release/signature/APK SHA-256 plus final full gates and affected device
checks; altered source/artifact, missing signing input, unsupported protocol or missing cleanup evidence fails its row.
**Gate/evidence:** complete current signed artifact/install/revoke/rollback evidence for M34; M38 supplies
later independent acceptance. M22 alone cannot close the full-owner release. No task completion claim substitutes for raw review evidence.
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
3. Use current owner authority plus a locally registered additional lifecycle capability and owner-side
   authorization/replay checks; no separate daily admin pairing or silent legacy upgrade;
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
under a named-file handoff. **Forbidden within this legacy/external task:** global Full, opaque local-reference serialization or
broad permission rewrites. This ceiling does not limit current owner Full through M30/M33.

1. Freeze exact target/session/task/source/revision/read-write extent and expiry using the current local
   permission owner; define local scope grant/revoke and one pending target proposal, not machine administration.
2. Keep all native authority/references server-local; remote device receives a bounded proposal identifier
   and sanitized description. Local owner mints a single-use receipt for the exact current binding.
3. Validate current device scope, unchanged source/target/revision and receipt before invoking the existing
   owner locally; consume atomically and prevent replay/approval at another task/session/device.
4. Test changed source/target, stale revision, expired/reused receipt, another device and scope revoke racing
   consume/delivery; assert no filesystem/task effect outside the granted target/read-write extent.
5. Add narrowly scoped UI/audit and independent authority review. This task is superseded for owner clients
   by M30–M33; retain it only for legacy/external exact-target approvals.

**Checks:** future authority/receipt/replay tests plus current native/permission regressions and full-source
type comparison; denial tests count effects, not merely hidden buttons or status codes.
**Gate/evidence:** every exact-binding/single-use/revoke negative passes; retain sanitized proposal/receipt
correlation and local owner review. No opaque reference or native capability crosses the wire.
**Rollback/inputs:** revoke approver scope/invalidate receipts and remove capability; keep normal chat and
local grants intact. Fresh accepted permission contract is required later; it never blocks the current owner workflow through M34/M38–M40.

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
## M30 — Single owner pairing and typed controller delegation

**Objective/prerequisites:** [M13](#m13) and [M08](#m08); implement owner semantics at the backend before
later emulator/phone integration. No later bridge or physical phone is a prerequisite.
**Allowed:** proposed/existing-at-M36 `nanobot/remote/{protocol,store,owner,ipc,client,pairing,auth,devices,http,entry,management,permissions}.py`,
`nanobot/config/schema.py`, narrow `nanobot/security/desktop_permissions.py` authority integration,
`tests/remote/test_{owner_delegation,owner_pairing,owner_permissions}.py` and existing affected authority tests.
Receive exact transferred paths/hashes before writes. **Forbidden:** second pairing/credential/store,
caller-supplied locality/native authority, generic IPC administration or runner/provider edits.

1. Grant one `owner` capability on new comparison-code-approved pairing with full-control/shared-history
   consent. Keep current legacy chat scopes valid without silent widening; re-pair or explicit server-local
   upgrade reconsents. Extend handshake/rotation/revoke consistently; never mint broad bootstrap tokens.
2. Define typed `DelegatedAdminOperation`: device list/rename/revoke/self-revoke, registered entry
   status/settings/disable, and permission grant/Full proposal/commit/release. Initial enable/re-enable,
   new pairing challenge/approval and scope addition remain admitted server-local operations. No generic
   command/config/PID/unit selector. Local-only exceptions are inventory rows, not hidden feature omissions.
3. Over selected-OS-user/role-authenticated IPC, forward the opaque device credential plus bounded operation/
   arguments/correlation/digest. Authoritative owner validates it and constructs `AuthenticatedDeviceContext`
   containing stable installation/device/principal, current scopes/record revision/expiry. Never accept
   caller assertions of identity, locality, native capability or scope. Revalidate revocation/scope before
   effect and result; internal owner handles bind device/action/digest/expiry and never reach the phone.
4. Owner authority authorizes the enumerated operation without changing network-local tests. Keep local
   setup/approval checks. RemoteAccessOwner returns typed device/operation-bound delegation admission;
   proposed `nanobot/remote/permissions.py` is the gateway-side adapter, not a new controller permission
   store. Invoke the existing selected server permission-owning process/adapter using that admission.
   It validates current source/session/task/revision, canonical target/extent, confirmation and expiry,
   and mints/consumes device-bound single-use native receipts **in that owning process**. Revalidate device
   revocation with RemoteAccessOwner at commit/delivery. No native reference/receipt crosses owner IPC or
   reaches the phone; the separate controller never inherits gateway/native task objects. Return safe
   projections/proposal IDs only, with no bypass of server filesystem/command policy.
5. Use deterministic barriers/clocks for delegated effects, revoke/scope changes, stale receipts and source/
   target/revision races. Publish permission/delegation vectors and ownership contract for M31/M37/M33.

**Checks:** backend/authority/native-admission tests: authorized owner IPC succeeds; missing/expired/revoked/
wrong-instance/legacy devices, wrong OS peer/role, unlisted operation, forged context/local/native fields,
remote setup/new-device approval and stale/reused/another-device receipt deny with effect count zero.
Positive Full/grant stays source/task/revision bound; ordinary local desktop/browser checks remain intact.
**Gate/evidence:** owner/legacy/local-only operation matrix, pairing/rotation/revoke vectors, permission
receipt/effect/race traces, exact schema/IPC changes and frozen same-environment type/regression evidence.
Backend fixtures suffice here; real current-owner emulator/phone checks belong to M14/M34.
**Rollback/inputs:** disable owner admission before incompatible rollback; retain installation/devices/
revocations/audit/sessions. Revoke only owned fixture identities, never silently downgrade or resurrect them.
Inputs are M36's current permission seams and M08 store contracts, not physical hardware/admin pairing.

<a id="m31"></a>
## M31 — Standard feature API and complete bounded native operation registry

**Objective/prerequisites:** [M30](#m30); expose every inventory action through one shared feature surface.
**Allowed:** `nanobot/remote/{edge,composition,protocol,client}.py`; existing
`nanobot/webui/{gateway_endpoint,gateway_tokens,gateway_services,ws_http,inbound_commands,workspaces,session_projection,media_gateway,media_api,settings_routes}.py`
and only additional route-handler files individually enumerated/frozen by M36 before dispatch;
matching tests. Transferred `mobile/contracts/{mobile-v1,operations-v1}.json`,
`mobile/src/native-broker.ts` and exact Go/Kotlin registry files recorded at M36. **Forbidden:** generic
raw-URL/localhost proxy, duplicate feature API/agent/session owner, exported credentials or inferred locality.

1. For every M36 row, map the existing standard HTTP route+method or `webui_request` action and media/
   binary/import/download/auth flow. Device control stays on the independent owner listener. Add no parallel
   mobile settings/session/skills API. Keep supported legacy shims through negotiated compatibility; remove
   only with a tested migration or accepted update-required cutoff, never silent legacy breakage.
   Separate existing-handler rows from planned adaptation rows with their producing task. Directory
   contract/registry entries use typed fixture vectors here; M37 implements and positively qualifies them.
2. Authenticate owner at standard HTTP/WS boundaries and attach verified device context. Existing feature
   handlers apply owner policy, current task/workspace/permission checks and M30 delegation; do not set
   `is_local`/`is_native` or manufacture native references. Local/browser auth stays unchanged. The owner
   bridge must explicitly deny server-local initial remote enable/re-enable, new-device challenge/approval
   and scope upgrade with the documented local-required error; do not expose them as ordinary settings or
   device actions.
3. Implement bounded named native registry entries for **all** required HTTP methods/WS actions and binary/
   auth flows: fixed origin/path templates, typed parameters, schemas, scope, time/byte/frame/media bounds,
   redirects/callback validation and cancellation. Reject undeclared methods/paths/substitutions/ports/headers.
   Use required provider-key transient write without returning existing secrets or persisting form payloads.
   Planned directory entries remain declared contract fixtures, disabled against an unsupported backend
   until M37 qualifies them; no release exception or live implementation PASS is inferred from fixtures.
4. Preserve M12 immutable identity/replay/current-auth behavior on every mutation and result, current media/
   range authorization, private permission/session projections and browser/desktop wire compatibility.
5. Extend row-by-row positive/negative route/native tests for existing handlers; planned later adaptations
   have typed contract evidence and explicit owners. Freeze registry/hash/version handoff before M14;
   M37 extends native positives and M34 requires all planned rows to be resolved before release.

**Checks:** backend, native registry and shared-client contract tests; owner positive for each inventory
existing-handler operation; legacy/scope/revoke/unknown/oversized/redirect and forged local/native negatives.
Planned directory operations require schema/registry fixture checks here and real backend positives in M37.
Check standard
settings/providers/models/apps/MCP/CLI/API/skills/automations/session/workspace actions and supported auth/
imports/downloads. Existing provider keys never appear in HTTP/WS/errors/media; transient new-key write works.
**Gate/evidence:** existing-handler owner/API/native positives and negatives pass; every planned later row
has a producing task and typed contract/registry fixtures. Retain limits/compatibility vectors, effect
counters and type/regression evidence. This gate does not require M37 directory implementation or final
full-inventory closure; those gates are M37 and M34 respectively. Emulator proof follows at M14/M16;
no real-phone requirement is smuggled into this backend prerequisite.
**Rollback/inputs:** disable owner ingress/registry version before incompatible rollback, retain current
device/revocation/audit state and legacy-compatible paths. Inputs are M36 exact actions/files and M30 IPC;
unimplemented required actions remain failed or explicitly accepted exceptions before parity claims.

<a id="m37"></a>
## M37 — Backend directory API and fresh grant-commit validation

**Objective/prerequisites:** [M31](#m31); deliver the server contract before its shared mobile UI consumer.
**Allowed:** proposed `nanobot/webui/workspace_directories.py`, existing `nanobot/webui/{workspaces,gateway_endpoint}.py`,
`nanobot/config/schema.py`, transferred `nanobot/remote/{permissions,protocol}.py`, narrow
`nanobot/security/desktop_permissions.py` validation, `tests/webui/test_workspace_directories.py`,
`tests/remote/test_directory_grants.py` and native registry/contract entries transferred from M31.
**Forbidden:** UI ownership, file contents/generic filesystem API, implicit filesystem grants or first-project selection.

1. Verify M36 route/convention collisions, then implement the plan's proposed standard API:
   `GET /api/workspace-directories/roots`; bounded `GET /api/workspace-directories/list` with root/path/
   cursor/limit; `POST /api/workspace-directories/resolve` with selected root/manual path/policy revision.
   Responses contain root labels/display paths, canonical relative breadcrumbs, directory names/opaque IDs,
   symlink/accessibility status, policy/directory revisions and next cursor only; no file content/listing.
2. Declare proposed `remote_access.directory_browser.roots` (`{id,label,path}`, camelCase aliases),
   default `[]`, maximum 32 roots. Empty config needs explicit setup; never infer `/`, home, native grants
   or first known project. Owner root changes use standard config write/server policy. Browsing visibility
   does not authorize filesystem/tools or constrain existing Full/manual-grant authority implicitly.
3. Canonicalize roots and each request via existing resolver/equivalent capability containment. Opaque
   browsing path IDs/cursors bind installation/root/policy/canonical directory identity and expire within
   five minutes. Manual relative or explicitly contained absolute input uses the selected root. Reject
   traversal/NUL/escape/out-of-root symlinks, stale IDs/root identity and unsupported identity proof.
4. Enforce limit 1–100 directories, at most 1,000 scanned dirents/request, 64 KiB response, two-second
   deadline, 4,096-byte path/depth 64 and no recursion. Cursor continuation remains bounded and rejects
   changed directory/policy. Emit sanitized 400/401/403/404/409/413/429/503 errors as defined in the plan.
5. Extend the existing M36-identified grant mutation, not a parallel grant API: re-resolve at commit under
   current policy/source/session/task/revision, compare canonical identity/target/extent and confirmation,
   and ask the existing permission-owning process for a fresh device/operation-bound receipt minted and
   consumed there, not in RemoteAccessOwner or across its IPC. Use handle/identity
   containment at issuance; reject rename/replace/symlink/policy races and never trust listing as authority.
6. Freeze request/response/error/native vectors and give M32 exact API/UI dependency handoff.

**Checks:** positive owner roots/list/manual/grant within configured policy; empty roots and permission-
denied directories are usable explicit states. Negative legacy/anonymous/wrong-instance/revoked requests,
manual traversal/symlink escapes, excessive entries/bytes/deadline, stale cursors/IDs and path replacement
between browse/confirm/commit fail with no grant. Concurrent source/task/revision/revoke/receipt races deny.
Existing desktop native picker/manual grants/Full and browser tests remain intact.
**Gate/evidence:** documented standard schemas and config defaults, deterministic finite-limit and
path-race/effect tests, native registry vectors, backend type/regression evidence and M32 handoff hashes.
**Rollback/inputs:** remove/disable only the new browsing capability; do not revoke unrelated workspace/
Full grants or erase sessions/identity. M36 permission-owner seam is required; no real server roots,
installed config, phone or live filesystem credentials are invented for synthetic tests.

<a id="m32"></a>
## M32 — Full shared owner UI and mobile workspace adaptation

**Objective/prerequisites:** [M17](#m17), [M19](#m19), [M31](#m31) and [M37](#m37); consume the frozen
backend/registry contract after explicit shared UI ownership transfer.
**Allowed:** M36-enumerated existing `webui/src` settings/providers/models/apps/skills/automations/sidebar/
composer/workspace components and tests, proposed `webui/src/components/ServerDirectoryBrowser.tsx`
and `webui/src/lib/server-directories.ts`. Exact existing paths/hashes are task inputs.
**Forbidden:** backend API/config/permission edits, mobile feature forks, native authority fabrication or raw fetch.

1. Use the shared Settings/Apps/Skills/Automations/provider/model/API-service pages and full payloads for
   owner devices. Remove legacy-only owner gating/model-only projection. Preserve legacy hidden/denied
   management and current server capabilities; no second mobile feature implementation. Hide or disable
   the server-local initial remote enable/re-enable, new-device challenge/approval and scope-upgrade
   actions, showing local recovery/enrollment guidance instead.
2. Consume M37 roots/list/resolve in a phone-friendly folder browser with breadcrumbs, directory/manual
   entry, bounded loading/error/permission/stale states and “Use this folder.” Native desktop picker remains
   desktop-only. Submit the verified target through the existing grant mutation and fresh confirmation.
3. Require explicit project choice or the current session's verified project. Empty selection stays empty;
   never choose the first known project. Preserve current workspace mode and Full access with source/task/
   revision/receipt-aware server behavior. UI state never itself grants authority.
4. Adapt every inventory action and supported auth/import/download flow at phone width. Restore full owner
   slash palette and existing supported restart/stop; advertise unsupported Start/service controls with the
   M19-recorded local recovery instruction (reverified at M35). Settings forms allow transient new
   provider-key input and clear it.
5. Preserve responsive composer/safe areas/Back/keyboard, temporary versus persistent disconnect semantics,
   browser/desktop behavior and instance binding through reconnect. Record evidence per inventory row.

**Checks:** WebUI component/route/contract tests and backend-phase emulator runner; positive all inventory
pages/actions, directory selection/grant/Full and key write. Negative legacy/admin visibility, stale/raced
path confirmation, first-project fallback, rejected/empty roots, overflow/error/loading, disconnected
temporary chat, unauthorized lifecycle and persisted/echoed provider input. Test browser/desktop regressions.
**Gate/evidence:** full row coverage in emulator/shared UI, phone-width screenshots/input traces and API/
registry/permission vectors. Real-phone acceptance follows M21/M34; it is not a prerequisite for this task.
**Rollback/inputs:** revert owned UI/adapter only or disable owner mutations if needed; retain authoritative
config/grants/history/devices. Inputs are M37 schemas and M36 exact page list; no installed runtime changes.

<a id="m33"></a>
## M33 — Owner confirmations, acting-device audit and secret safety

**Objective/prerequisites:** [M32](#m32); all safeguards precede integrated qualification.
**Allowed:** transferred shared confirmation/provider/device UI and tests; proposed `nanobot/remote/audit.py`,
`nanobot/remote/{store,owner,management,permissions,auth,devices}.py`, M36-enumerated standard mutation
audit hooks and `nanobot/webui/settings_{routes,contracts,services,system}.py` redaction; closest tests.
**Forbidden:** silent destructive effects, provider-secret reads/logs/persistence, bearer/native authority export.

1. Confirm destructive/sensitive inventory actions: session delete, effectful skill/MCP/CLI install/update/
   removal, automation create/run/delete, grant/Full, device revoke, remote disable and supported runtime
   restart/stop. Bind confirmation to acting device/action/target/digest/correlation/expiry and current
   source/task/revision where applicable. Reject stale/reused/cross-device receipts and changed targets.
2. Write bounded sanitized audit records with stable installation ID **and acting device ID/principal**,
   operation correlation, action, sanitized target, outcome and timestamp. Record denial/failure/replay
   outcomes without a duplicate effect. Bound metadata sizes/retention policy without erasing revocation
   history. Never log tokens/provider keys/private contents/raw payloads/native references or receipts.
3. Existing provider secrets remain write-only across all HTTP/WS/error/export/log surfaces. A newly typed
   key may pass transiently from the shared form to the standard write endpoint, then is cleared; exclude
   storage/autosave/analytics/replay caches/echo and redact exceptions. Durable app credentials stay native.
4. Keep self-revoke available. Revalidate current revocation/scope before all effects/results, retire active
   sockets/handles within the bound, retain short TTL/atomic rotation and show local recovery before entry-off.
   Optional later step-up does not block this accepted scope.
5. Run two-device, revoke/confirm/dispatch/delivery races and audit tamper/failure-mode tests; fail closed
   for mutations requiring durable audit if audit persistence fails. Publish safeguard matrix for M35.

**Checks:** backend/WebUI/native privacy/authority tests; two owner devices yield distinct acting principals
under one installation and correlated outcomes. Missing/stale confirmations, revoked device, bad receipts,
audit write failure and secret echoes deny; positive transient provider write, confirmed owner action and
self-revoke pass. Current permissions, legacy denies, TTL/rotation and browser/desktop regressions remain.
**Gate/evidence:** action-by-action confirmation/audit/redaction/current-auth matrix, sanitized two-device
records, negative effect counts/race timing, full-source type comparison and exact safeguard hashes.
Physical live revoke follows M21/M34, not an early prerequisite.
**Rollback/inputs:** disable owner mutations/ingress rather than weaken safeguards; retain all durable
audit/devices/revocations/sessions. Inputs are frozen action owners and M30 permission receipts; no new keys.

<a id="m35"></a>
## M35 — Isolated desktop-equivalent instance integration

**Objective/prerequisites:** [M19](#m19), [M32](#m32) and [M33](#m33); integrate current owner behavior
on a separately selected isolated instance before full candidate/device/release qualification.
**Allowed:** isolated install/launcher/config/service/mapping runbook, synthetic development instance and
`mobile/evidence/<run-id>/M35/`; scripts/tests within recorded verification ownership. Fixes return to
product owners. **Forbidden:** installed staging config/data/environment/credentials as fixtures, actual
installed promotion, a second writer per instance or deleting durable selected-instance owner records.

1. Freeze current candidate server package/commit, bundled WebUI hash, internal app/hash, registry/protocol/
   schema and exact isolated environment/launcher/config/data/instance identity. Use synthetic provider/
   workspace/history fixtures; no installed credentials/sessions copied. M22 later supplies the exact signed
   artifact; repeat affected M35 rows after any candidate change and bind the final manifest at M34.
2. Install/run that candidate only in the isolated environment, with one gateway, one independent owner and
   one canonical data set per instance. Record owner user-service resource/command/config/socket/listener and
   selected gateway lifecycle owner/launcher; do not adopt a foreign or service-owned process as desktop-owned.
3. With concrete isolated service/network opt-in, map one private HTTPS origin's standard API/WS/media paths
   to their actual loopback gateway listener(s), and `/remote/v1/*` auth/status/refresh/device-control directly
   to the independent owner. Multiple owned mappings are valid. Preserve registered prefixes/segment
   boundaries/longest match/query/encoded-path/WS Upgrade; no control route depends on a running gateway.
4. Pair emulator once as owner through local comparison approval; prove desktop/browser and phone see the
   same installation, canonical sessions/models/tools/settings/skills/automations/workspace/Full and every
   inventory action/adaptation. Exercise legacy ceiling, device management/local-only enrollment exceptions,
   transient key write, audit/revoke, temporary disposal versus persistent detach and owner restart.
5. Stop/kill the isolated gateway through supported ownership semantics; prove owner auth/status/refresh/
   device revoke independently reachable and no second gateway starts. Use a separate still-authorized fixture
   device for refresh after revoking another. Advertise only existing supported restart/stop. Remote Start
   stays unsupported unless separately scoped; record the exact tested local launcher/recovery command
   and display it to the phone. Preserve guarded Quit and unsupported service-manager refusals.
6. Freeze the topology/config/launcher/environment/route evidence for M20 and later M34/M38 review; rehearse
   owned-resource teardown and retained-state restart. This task does not switch normal desktop installation.

**Checks:** backend-phase emulator/browser/desktop integration and isolated service/network rehearsal;
positive full inventory/same-instance behavior; negative stopped-gateway route coupling, prefix mismatch,
foreign/duplicate runtime, stale device/scope/native assertions, legacy administration and data-copy leakage.
Record each listener/mapping separately; never require exactly one Serve mapping.
**Gate/evidence:** isolated current full-owner instance matrix with exact component/config/launcher hashes,
one-writer counts, gateway-absent auth/status/refresh/revoke traces and honest local recovery. No installed
staging activation claim. Missing isolated network/service input leaves only its live criteria PENDING.
**Rollback/inputs:** deny ingress, remove only owned test mappings/supervisor resources, stop recorded
test processes and delete only explicitly disposable fixture data. Retain non-disposable installation/
device/revocation/audit/session state. An incompatible downgrade keeps ingress disabled until protocol/
store compatibility passes; never restore stale credentials. Actual isolated origin/service/launcher
and fixture enrollment are real inputs; no installed production opt-in is consumed here.

<a id="m34"></a>
## M34 — Complete owner parity, physical qualification and release bundle

**Objective/prerequisites:** [M22](#m22), [M32](#m32), [M33](#m33) and [M35](#m35); close all current
owner release evidence after isolated integration and current candidate physical/signing checks.
**Allowed:** inventory status/evidence/release runbook and verification scripts/tests in handed-off paths;
product fixes return to their task owners. **Forbidden:** parity by representative checks, historical
PASS reuse without applicability, installed activation or independent author self-acceptance.

1. Check every M36 inventory row's owner/legacy auth, native transport/limits, shared UI, mobile adaptation
   and positive/negative evidence on the exact candidate. Resolve unsupported rows by implementation or
   explicit product acceptance before a parity claim; never hide a category or treat full suites as coverage.
2. Link current M20 full Python/WebUI/native/build/lint/strict-regression evidence, M21 actual physical
   networking/custody/power/media/accessibility and M22 signing/reproducibility/install/update/reinstall/
   different-key rejection/revoke/rollback. Repeat affected rows on the exact final signed APK.
3. Physically verify all inventory categories/actions, including settings/providers/models, Apps/MCP/CLI/
   API service, local/custom/installed skills, automations, temporary/persistent/session actions, workspace
   mode/grants/Full, owner device/entry management and supported lifecycle/auth/import/download flows.
   Check one pairing, update persistence, sign-in/refresh uncertainty and local-only enrollment exceptions.
4. Freeze final server package+commit, bundled WebUI SHA-256, APK/app version/certificate fingerprint,
   protocol/native registry/config/data-schema versions, candidate config/environment/launcher and all
   evidence hashes. No later source/artifact change may inherit this bundle's results automatically.
5. Prepare a read-only promotion proposal for the actual operator-selected installed target when supplied:
   prior/desired package/environment/executable/launcher/config/data identity/supervisor/mappings, compatible
   data/protocol/backup/rollback and exact local recovery. Never use installed data as test fixtures or
   mutate it here. Missing target leaves promotion-specific evidence PENDING while isolated qualification
   remains usable; M38 must review the concrete proposal before M39 may change anything.

**Checks:** deterministic complete row/evidence/hash cross-check plus final full gates and affected real-
phone checks. Missing row/command output, stale source/signature, skipped required live criterion, uncovered
local-only exception or inconsistent server/WebUI/APK/registry fails the corresponding gate.
**Gate/evidence:** complete current isolated owner release bundle with binary row status and exact final
versions/hashes, ready for independent M38 review. This is evidence completion, not installed activation
or final human/independent acceptance.
**Rollback/inputs:** preserve candidate/durable state; disable test ingress and revoke disposable identities
only. Physical Android/signing/enrollment are real required release inputs; actual target selection is
required for a reviewed promotion proposal, never guessed from an example staging config.

<a id="m38"></a>
## M38 — Independent integrated review before promotion

**Objective/prerequisites:** [M34](#m34); independently review the actual current candidate and proposed
installed transition after qualification, never after experimental installed activation.
**Allowed:** read-only actual diff/source/artifact/evidence inspection by the assigned independent reviewer,
sanitized review report and acceptance matrix. **Forbidden:** product edits, runtime/install/config/mapping
changes, author self-review acceptance or waiving missing qualification.

1. Verify final manifest hashes and inspect all M36 inventory rows, implementation/evidence, negative
   authority/replay/transport/media/current-revoke paths and desktop/browser compatibility.
2. Review owner IPC/delegation/local-only exceptions, permission receipts/races, acting-device audit,
   credential/provider-input redaction, stopped-gateway control topology/supervisor and durable rollback.
3. Check physical Android/signing/update/reproducibility and full required automated/type evidence.
   Historical reports do not fill fresh gaps. Issue PASS/CORRECT/REJECT per criterion with concrete evidence.
4. Independently review the concrete M34 promotion proposal when the actual target is supplied: exact
   server/WebUI/mobile versions, config/data/environment/launcher/service/mappings, single-writer transition,
   compatibility, retained revocations/new writes and rollback. This is read-only preparation before M39.
   Missing target/proposal leaves **activation review PENDING**, not implicit permission.
5. Return corrections to named owners, requalify affected rows and re-review exact revised hashes before
   acceptance. Changes after PASS invalidate affected review; no review depends on M39 mutation.

**Checks:** independent hash/coverage/negative-path/diff audit; stale/missing artifact, unsupported operation
without accepted exception, unsafe promotion/rollback or unverified external criterion cannot receive PASS.
**Gate/evidence:** isolated release review PASS bound to exact hashes. M39 additionally requires independent
PASS for its concrete installed transition; until then activation is PENDING even if isolated release is accepted.
**Rollback/inputs:** review rejection leaves installed state untouched and test ingress disabled as needed.
Inputs are complete raw M34 evidence and a genuinely selected target for activation review. Actual promotion
opt-in is separate; this task neither installs anything nor grants external permission.

<a id="m39"></a>
## M39 — Controlled reviewed installed promotion

**Objective/prerequisites:** [M38](#m38); require PASS for both exact release and concrete installed
transition, actual operator-selected target and concrete install/config/service/network opt-ins.
**Allowed:** only the reviewed installation mechanism, target package/environment/launcher/config and
owned supervisor/mapping resources enumerated in the M38-accepted transition; sanitized rollout evidence.
**Forbidden:** experimental bundle copy, unreviewed versions/config drift, staging source edits, global
Tailscale changes, competing gateway/owner writers or invented target/credentials/privileges.

1. Before mutation, compare actual installed version/paths/config/data identity/services/mappings to reviewed
   prior-state hashes. Present the exact desired transition, impact, backups, compatibility and rollback for
   any missing real opt-in. A changed target/candidate returns to read-only proposal/review, not blind execution.
2. Verify backup/recovery without treating installed state as fixtures. Preserve installation identity,
   devices/revocations/audit/sessions and new writes. Disable ingress/quiesce, stop the old authoritative
   gateway through its actual owner, and prevent old launcher/supervisor from restarting a second writer.
3. Install the exact reviewed server package/commit and matching bundled WebUI into the selected environment
   through its chosen local installer. Record actual artifact hashes/executable/version; config/Serve changes
   alone are not an upgrade. Transition launcher/environment/config schema with explicit compatible data
   handling and preserved server credential references, never leaking secrets in evidence.
4. Start exactly one selected-instance owner under the reviewed independent supervisor and one gateway
   under its existing lifecycle owner. Verify installation/protocol/store compatibility locally before
   applying the reviewed data/control mappings and enabling device admission. Retain local desktop access.
5. Record prior/desired/observed versions/config/environment/launcher/service/mappings and identity, then
   hand off M40. Stop/retire only owned temporary test gateway/enrollment/entry resources; retain their
   durable audit/revocation records unless the entire fixture was explicitly disposable.

**Checks:** deterministic installed artifact/version/identity/ownership/readiness comparison; negative stale
prior-state hash, incompatible store/protocol, duplicate runtime, wrong bundled WebUI or mapping conflict
halts with ingress off. Do not call activation verified until M40 tests the actual installed target.
**Gate/evidence:** exact reviewed build and transition installed with one runtime/owner, durable identity/
revocations/sessions retained, compatible local readiness and precise observed mapping/service record.
Missing external target/opt-in leaves promotion PENDING; isolated release acceptance remains valid.
**Rollback/inputs:** disable ingress first; stop only the selected new runtime; restore the reviewed previous
package/environment/launcher/config only if compatible with current durable data. Preserve/reconcile new
writes and all revocations/audit; never blind-restore old owner records. If old backend cannot honor current
protocol/store, keep remote ingress disabled and use the recorded local recovery. Remove only owned
resources, never installed owner state. Actual installation target/mechanism/privileges and opt-in are inputs.

<a id="m40"></a>
## M40 — Postactivation actual same-instance verification

**Objective/prerequisites:** [M39](#m39); verify the actual promoted target with the qualified signed
phone and operator's non-destructive/carefully bounded check selection.
**Allowed:** installed-target version/status and approved fixture-operation/device checks, sanitized evidence/
recovery runbook; exact reversible operations listed in the reviewed transition. **Forbidden:** broad user-
data mutation/deletion, new features/config changes, invented test credentials or retroactive promotion approval.

1. Compare observed server package/WebUI/APK/protocol/registry/config/environment/launcher/supervisor hashes
   to M38/M39. Verify desktop and phone show the same stable installation and canonical history/model/tools/
   workspace policy; no temporary test server remains the phone's product target.
2. Verify standard owner feature/API/auth/media and legacy ceiling on bounded synthetic/explicitly selected
   data, including provider write redaction, grant/Full confirmation/audit and current device revocation.
   Use the complete isolated matrix as qualification; record which live rows are non-destructive probes and
   do not claim unexecuted installed mutations as PASS.
3. Rehearse supported restart/stop only within the concrete opted-in window and existing owner semantics.
   With gateway absent, prove owner status/auth/refresh/device control remain independently reachable;
   remote Start stays honestly unsupported and the exact local recovery command works. Restore intended
   gateway state through that local owner without a second writer; keep guarded Quit behavior.
4. Verify phone update/reconnect, installation/device persistence and self-revoke/re-pair through server-local
   comparison approval using designated test device records. Verify entry-off warning/recovery only when
   explicitly included in the transition; pending live criteria stay PENDING rather than fabricated.
5. Confirm owned temporary mappings/processes/enrollments removed, durable installation/device/revoke/audit/
   sessions retained and final observed versions/routes/services documented. Trigger reviewed rollback on
   failed safety/compatibility checks rather than continuing remote admission.

**Checks:** same-instance identity/history/policy comparisons, approved live operation/effect/audit and
gateway-absent control probes; wrong instance/stale build, resurrected device, local-only remote approval,
duplicate writer or control-route coupling fails. Do not broaden live tests into unapproved destructive work.
**Gate/evidence:** actual installed same-instance and recovery matrix PASS on exact reviewed versions with
honest pending external/live rows. Only complete required live rows close the installed-owner finish;
signed/isolated acceptance alone is not this result.
**Rollback/inputs:** invoke M39's reviewed ingress-first compatible rollback, preserve all durable records/
new writes and report actual observed state. Inputs are the qualified physical Android/signing identity,
actual installed target, designated test device/data and concrete live-check window/opt-in.
