# Fedora Tauri desktop implementation plan

2026-10-23 · source baseline `staging` at
`fdc36a9e325b9055023a0afd56f08d6ed1aca2b0`.

This is an implementation specification, not implementation or release acceptance.
Only this document was created. No dependencies were installed, services changed,
providers exercised, or runtime tests run for this planning task. This plan is
complete; implementation, native feasibility and release qualification remain pending.

## Accepted scope and current evidence

Build a small Tauri 2 shell for Fedora KDE Plasma on Wayland around the existing
React WebUI and Python gateway. The normal entry is a desktop icon; chat appears
inside the app. Browser and headless use continue to work. V1 distributes a thin
RPM over an explicitly selected, already installed compatible Python backend.

Accepted choices: retain and hide the existing chat webview; keep one selected
personal gateway persistent; restore the same window/conversation from the tray;
make **Quit Nanobot** stop that managed gateway before the desktop exits. Hiding
preserves drafts, attachments and streams, but does not promise renderer memory
release. Scheduling and process continuations keep their existing Python owners.

Deferred: renderer destruction/recreation, bundled Python, login autostart,
unattended updates, native folder/file tooling, Windows/macOS packaging, Qt/C++
UI, a Rust agent harness, scheduler changes, and marketing production cutover.
There is no new durable ownership database, agent loop, tool policy or business
logic. No formal migration-stage approval is created by this plan.

Source inspection established these implementation seams:

| Evidence inspected | Consequence |
| --- | --- |
| `webui/package.json`, `webui/vite.config.ts`, `pyproject.toml` | React 18/Vite/TypeScript already builds into `nanobot/web/dist`, included in Python packages. Normal desktop use needs no Vite, Node or Bun server. |
| `webui/src/lib/{bootstrap,api,nanobot-client,runtime}.ts`, `webui/src/App.tsx` | Bootstrap/reads use the gateway origin; mutations use authenticated WS; one client multiplexes/reattaches chats. Fragment secrets are removed and the existing app saves bootstrap secrets in localStorage. Native host hooks are groundwork, not a desktop package. |
| `nanobot/cli/{webui,webui_support,tui_launcher,gateway}.py` | Local clients already share a selected `GatewayInstance` and on-demand leases. The last lease can stop an on-demand gateway. Existing launch helpers include browser/TTY/setup behavior unsuitable for direct desktop reuse. |
| `nanobot/gateway/runtime.py`, `nanobot/process_runtime.py` | Canonical config/workspace selectors, PID birth identity, background launch and transition/lifecycle/client locks exist. `start_background` promotes lifetime before classifying an existing process. POSIX stop signals a process group and escalates to SIGKILL after the grace period. Neither operation should be called blindly by the shell. |
| `nanobot/migration_guard.py`, `runtime_composition.py`, startup-admission tests | Legacy no-argument admission still always raises. Normal config-aware preflight and receipt admission are separate supported seams; preserve their refusals and the administrative fence. |
| `nanobot/cli/gateway_runtime.py`, `nanobot/process_completion.py`, scheduled-execution/shutdown tests | Gateway shutdown closes cron, continuations and composition. The foreground state claim can be released before final cleanup; state-file absence alone is not process-exit proof. Interrupted effects retain typed uncertainty. |
| `webui/src/hooks/useVoiceRecorder.ts` | Cleanup currently depends on unmount. A hidden retained webview needs explicit capture cleanup, including a pending microphone-permission request. |
| `tests/gateway/test_runtime.py`, both startup-admission modules, CLI WebUI/TUI/gateway tests, WebUI smoke/terminal tests | Existing tests exercise instance matching, lease promotion, replacement races, degraded readiness, unknown-identity refusal, admission ordering and authenticated capability probes. These are regression anchors, not evidence that desktop behavior exists. |

The [prototype integration](assistant-prototype-integration.md) and
[validation record](assistant-prototype-validation.md) describe prior validated
backend behavior and commands. Their counts are historical evidence; no fresh
desktop, full-suite or strict-check PASS is claimed here.

Read-only environment probes found Fedora 44 x86_64, WebKitGTK 4.1 runtime
2.54.1 and GTK3 3.24.52. Rust/Cargo and `uv` are not on this shell's PATH; Bun
and pkg-config are. WebKitGTK, appindicator, librsvg and libxdo development
packages are missing; OpenSSL development files are installed. The supplied
contract identifies the current graphical session as KDE/Wayland.

## User behavior and states

Tray menu: a disabled **Starting / Running / Unavailable / Stopping** status row,
**Open Chat**, contextual **Retry**, and **Quit Nanobot**. Errors also have a
reachable window with a sanitized reason and selected config/interpreter paths.
Do not rely on tooltips for status on Linux.

| Trigger/state | Required result |
| --- | --- |
| First launch, no saved selection | Show a small setup/status window to select an existing config and absolute Python interpreter. No implicit fallback or backend start. Remember only these selectors and optional workspace override in private user preferences. |
| Launch with valid selection | Show Starting, perform the compatibility/admission checks below, atomically start or adopt the selected managed background gateway, and wait for readiness. Open the gateway's chat origin inside one chat webview. |
| Ready, close window | Intercept close, stop microphone capture, then hide the same webview. Host, adapter, WS and persistent gateway stay alive. Do not release lifetime, unmount chat or reload the page. |
| Tray menu → Open Chat / repeated icon launch | Show/unminimize and request focus on the existing window. Preserve its route, conversation, draft, attachments and stream. Wayland focus denial must still leave a visible taskbar-accessible window. |
| No tray host, tray creation failure or host disappearance | Keep/show the window and retain a taskbar control; close minimizes instead of leaving the app unreachable. A successful tray constructor is not proof a tray host exists. Keep Quit available in a native window/application menu. |
| Live process, unavailable listener | Show Unavailable, retain chat/draft state, and poll bounded readiness without starting a replacement. Existing channel recovery and WS reconnect may restore Running. |
| Gateway or adapter crash | Show Unavailable and the reason. Reconcile identity on explicit Retry; a live verified gateway is reattached, a proven absent gateway may be started after preflight. No blanket automatic gateway restart or effect replay. |
| Host crash | A persistent gateway may remain. Relaunch verifies the selected instance and reattaches; it must not create another writer. Unsent in-memory drafts are not promised to survive a crash. |
| Quit Nanobot | Enter Stopping, reject additional starts, gracefully stop the verified selected gateway, confirm process exit, close the adapter and exit the desktop. Other attached browser/TUI clients lose that backend connection. |
| Stop timeout / replaced or unverifiable process | Restore/show an error window, retain tray and Retry, and report that shutdown is unconfirmed. Do not report success, clear ownership evidence, force-kill an arbitrary group or block the UI indefinitely. |
| Repeated launch during Stopping | Focus the stopping window; explain that launch is refused until shutdown finishes. Do not queue an automatic restart. A launch after completed exit can start normally. |

If no gateway was admitted, Quit exits the desktop only; the conflict/setup
screen must make clear that any external backend remains unmanaged and running.

Linux guarantees **menu → Open Chat**. Tauri does not emit direct tray mouse
events on Linux, and its left-click-menu setting is unsupported there; do not
promise direct one-click activation. Supply an icon and real menu, whose content
can be updated without replacing the Linux menu. See the official
[tray guide](https://v2.tauri.app/learn/system-tray/) and
[TrayIconBuilder reference](https://docs.rs/tauri/latest/tauri/tray/struct.TrayIconBuilder.html).

## Ownership and narrow interfaces

| Owner | Responsibilities and boundary |
| --- | --- |
| Rust `desktop/src-tauri` host | Single-instance activation, windows/tray, private selector preferences, explicit interpreter launch, pipe supervision, readiness display, constrained external opener and quit orchestration. No direct gateway signals, lock-file editing, tool/file permissions or agent/scheduler logic. |
| New Python `nanobot/desktop` edge adapter | Strictly load the selected config, handshake, preflight, invoke public gateway lifecycle operations, bounded readiness/bootstrap, and typed/sanitized pipe responses. It is a client/control adapter, not a second gateway or service manager. |
| Existing `GatewayRuntime` / process owner | One canonical instance, locking, atomic persistence, captured process identity, classified adoption, signal validation and shutdown confirmation. Consumers must not reach into private lock/state helpers. |
| React and existing Python gateway | Existing chat, uploads, streams, authentication, HTTP/WS routing, durable sessions, tools, cron and continuation ownership. React gains only hide/visibility recording cleanup. |

Use the configured absolute interpreter directly, with an argument array such as
`/absolute/venv/bin/python -m nanobot.desktop`; never a login shell, PATH search
or per-launch install. Preserve a selected virtualenv entrypoint rather than
resolving its Python symlink into an unrelated system environment. Canonicalize
config/workspace using `GatewayInstance.resolve`. V1 supports one selected
instance per desktop host; changing selectors requires a completed Quit first.

One private inherited stdin/stdout pipe carries newline-delimited JSON. Recommend
protocol v1 with request IDs and only `hello`, `start`, `status`, `bootstrap`,
`stop`; no arbitrary method, command, path-operation or eval transport. Bound
frames to 64 KiB and reject malformed/oversize/unknown messages. Serialize
lifecycle requests; use monotonic deadlines and a background worker so neither
pipe reads nor gateway locks block the native event loop. EOF releases the
adapter's client lease, not persistent backend lifetime. Reserve stdout for the
protocol; sanitize and drain bounded stderr without forwarding credentials.

`hello` must complete within 5 seconds and return a typed protocol version,
backend package version, effective interpreter and required lifecycle/bootstrap
capabilities. Validate against the RPM's committed compatibility manifest before
any start/adoption. Initially support only the exact backend version(s) qualified
with that desktop release plus protocol v1; do not invent a version pin now or
infer compatibility from the current `0.3.0` label alone. Missing module,
unsupported protocol/capability or interpreter mismatch shows a repairable error
and never falls back to another Python. The running gateway must also prove the
required backend capability; a new adapter cannot attest an old live process.

## Selected-instance admission, persistence and stop

Implementation recommendation: add small typed public desktop operations on
`GatewayRuntime`, rather than composing private helpers in Rust or the adapter.
Retain the current state files and lock order: transition → lifecycle → clients.
Finite lock acquisition failure returns Busy/Retry. Do not hold the lifecycle
or memory-receipt mutation lock while waiting for child shutdown or launching
under a receipt mutation context.

1. Require an existing readable explicit config. Validate the selected workspace,
   runtime-data/session roots, loopback WebSocket/management binds, static assets
   and auth settings. Invalid config/environment substitution must fail instead
   of silently loading defaults or `~/.nanobot`. Desktop launch does not run the
   interactive `nanobot webui` command, create config, regenerate credentials,
   change provider/channel defaults or build missing assets. Show setup guidance
   when the existing backend is not prepared.
2. Use the config-aware `runtime_preflight(config, sessions_root=...)` and receipt
   admission before mutation, and recheck at the lifecycle handoff. The normal
   gateway child still performs its own admission. If a bounded setup write ever
   becomes necessary, use `admitted_startup_mutation` only around that write,
   release it, then recheck before spawn. Do not remove/patch
   `require_runtime_ready`, introduce an environment bypass, or unblock legacy
   background/admin/service paths as part of desktop work.
3. Under the existing runtime locks, classify and verify before **either**
   promoting persistence **or** authorizing stop. Match canonical selectors,
   recorded child command/interpreter, same-user PID birth identity and managed
   background provenance. On Linux require the expected detached process-group
   and session leadership; reject shared groups. Foreground gateways (including
   the existing service installer's foreground command), known supervised
   gateways, unknown launch modes/identities, and incompatible legacy records
   are conflicts. Leave their lifetime/state/process unchanged. An unrelated
   listener on the configured ports is a conflict, never attachment authority.
4. For a new gateway, establish explicit/persistent lifetime and register the
   desktop lease in the same locked transaction before spawn; for a verified
   managed on-demand gateway, promote in that transaction before another lease
   can trigger last-client shutdown. If shutdown already committed, wait within
   the deadline and reconcile before retry. Do not first start on-demand and
   promote after readiness. Reuse the existing detached background mechanism.
5. Return an in-memory selected-instance handle containing selectors, PID/birth
   identity and process-group identity. Keep no second ownership ledger. Readiness
   requires that exact live handle, management `ready=true`, available static
   assets and successful authenticated `/webui/terminal` probing at the exact
   configured origin. Use its existing stable `gatewayId` without issuing new
   tokens on every poll. To bind that endpoint to the selected process, publish
   the WebSocket owner's non-secret gateway ID and backend capability/version
   into the **existing** runtime state through an owner method that checks its
   current PID identity. Compare the probe to that record; missing/mismatched
   identity is Unavailable/conflict. This is a small edge addition, not a new
   frontend protocol or HTTP endpoint. Legacy records remain usable by their
   existing clients but cannot be adopted by desktop without this proof.
6. Recommend a 30-second startup/readiness deadline, 1-second status polling,
   and per-HTTP-probe timeout no longer than 1 second with bounded response
   bodies/no proxy or redirect following. Timeout presents Unavailable/Retry.
   Polling may recover an already live listener; only explicit Retry can start
   a proven absent backend. Do not restart a live gateway merely for degradation.
7. Quit calls the runtime owner's verified stop with the captured handle.
   Recheck classification and birth/group identity under the transition and
   lifecycle locks, then allow child cleanup to take the lifecycle lock while
   the transition stays serialized. Revalidate at signal boundaries. Recommend
   a desktop-only **graceful** stop option: SIGTERM through the verified existing
   process owner, 20-second grace, no automatic SIGKILL; preserve current CLI
   defaults. Await actual captured-process exit, not state-file disappearance
   or a generic stale-state result. A changed handle or replacement state is a
   conflict requiring reconciliation, never permission to signal the replacement.
8. Bound stop/pipe completion to 25 seconds including lock/protocol overhead.
   A timeout keeps the desktop controls alive; Retry revalidates the same target.
   If the adapter dies after a stop request, reconnect/reprobe before deciding
   whether it completed; do not blindly replay it. Clear only the verified
   exited instance's lease/state through the owner, preserving a replacement.

Quit uses existing shutdown semantics: stop admitting work, close cron and
continuation tasks, cancel/join runtime tasks and close composition. It does not
promise to finish every active turn, roll back an external tool or resume every
process. Retain journals, receipts and unknown/interrupted outcomes for existing
recovery. Do not replay an uncertain delivery or continuation from the desktop.

## Webview, authentication and microphone boundary

Load `http://<configured-loopback-host>:<configured-port>/` after readiness,
including the exact origin's scheme/host/port; do not substitute `localhost`
for `127.0.0.1`. Validate that bootstrap `ws_url` remains the selected loopback
gateway; refuse a remote override in desktop v1. Do not serve chat from
`file://` or `tauri://`, proxy the whole frontend through Rust, or duplicate WS
mutations. Keep `runtime_surface=browser`; do not install `window.nanobotHost`
or claim picker/restart/diagnostics capabilities.

The adapter reads the existing bootstrap secret and returns it only over its
private pipe. Rust builds the initial fragment URL in memory for the webview;
the current frontend consumes/removes the fragment. No secret goes into process
arguments, `.desktop` Exec fields, environment propagation, preferences, URLs
sent to an OS opener, logs, checked-in config or diagnostics. Use typed error
codes instead of dumping config, full URLs, pipe frames or exceptions. Poll
`/webui/terminal` with its authenticated header; fetch bootstrap only when needed.
The current WebUI localStorage credential behavior remains browser-owned; keep
the webview profile private and isolated to the selected instance, and exclude
it from diagnostics. Do not describe it as an OS keychain or memory-only store.

Register single-instance first and perform no adapter/start side effects before
primary-instance selection. Subsequent-launch arguments are untrusted: activation
only, with no silent selector change. The plugin's Linux implementation uses the
session bus; absence/failure must fail visibly before starting a second backend.
See [single-instance documentation](https://v2.tauri.app/plugin/single-instance/).

Keep gateway pages without remote Tauri IPC capabilities. A minimal bundled
setup/status page may receive only its fixed selection/retry/quit commands;
scope them to that local window. No shell/fs/process command permissions for
chat, no wildcard remote capability, no arbitrary native bridge. Tauri explicitly
requires opting remote pages into API access; retain the default boundary.
[Capabilities documentation](https://v2.tauri.app/security/capabilities/).

Use native navigation/new-window handlers to keep only selected-origin chat
navigation in the webview. User-activated HTTP(S) links and provider OAuth URLs
open via a constrained system opener; validate scheme/URL, reject embedded
credentials and executable/file/data/javascript/custom schemes, deny remote
replacement/popups and do not follow redirects with native credentials. Browser
escape links are credential-free. Preserve normal browser OAuth/callback behavior
and test representative providers with local fixtures. Do not assume Tauri's
bundled-asset CSP injector controls gateway-served HTML; retain its existing
server headers. See the
[webview builder reference](https://docs.rs/tauri/latest/tauri/webview/struct.WebviewWindowBuilder.html).

Before hiding, send one fixed native-to-page visibility notification with no
generic page-to-native privilege; also handle browser `visibilitychange`.
The recording hook must clear hold/max timers and waveform/audio contexts, stop
tracks immediately, and fence a late `getUserMedia` resolution so it cannot start
hidden capture. Finalize already captured audio through the current transcription
path into the unsent draft; a short/failed recording yields the existing visible
error on reopen. Never send the draft automatically or resume capture on show.
Do not cancel an existing chat stream or upload to achieve capture cleanup.
The Fedora probe must prove notification execution before hide; if that cannot
be established, keep the window reachable until cleanup is confirmed.

## Ordered work packages

Each package owns its full overlapping boundary, including tests. Preserve other
edits. One writer owns Python lifecycle/adapter work; shell and frontend packages
can run in parallel only after the pipe/visibility contracts are fixed, with
disjoint files. Integration/packaging follows their join.

| Package | Likely files and dependencies | Binary completion condition |
| --- | --- | --- |
| P1: Fedora feasibility | No product implementation until prerequisites are available. A temporary minimal Tauri probe; record results with the implementation validation artifacts. | Release-mode probe proves tray menu/show/hide/focus, missing-host fallback, session-bus single instance, selected-origin bootstrap, no remote IPC, constrained navigation, microphone notification ordering and WebKit compatibility. Any failure gets a concrete bounded fix or keeps the relevant feature/release blocked. Direct tray click is optional only if a supported path is proved. |
| P2: Python edge/lifecycle | New `nanobot/desktop/{__init__,__main__,protocol,adapter}.py`; narrow owner edits in `nanobot/gateway/runtime.py`, `nanobot/process_runtime.py`, `nanobot/cli/gateway_runtime.py` and the WebSocket edge that publishes its existing gateway ID. New `tests/desktop/`; extend owning gateway/admission/terminal tests. Depends on accepted contract; independent of graphical code. | Typed pipe and compatibility parsing, clean config-aware admission, atomic persistent adoption/start, readiness binding and graceful verified stop pass all lifecycle cases below. Existing CLI/TUI/browser leases and administrative denials remain intact. No core agent/tool changes. |
| P3: Native shell | New `desktop/src-tauri/{Cargo.toml,Cargo.lock,build.rs,tauri.conf.json,src/*,capabilities/*,icons/*}` and a small bundled setup/status asset. Stable compatible Tauri 2 dependencies chosen in implementation. Depends on P1 and P2 protocol. | One primary host/adapter/chat window; exact lifecycle states/menu, recoverable errors, selection storage, constrained opener, IPC refusal and bounded Quit pass native unit/probe tests. No frontend rewrite or Python bundled in RPM. |
| P4: Recording hide cleanup | `webui/src/hooks/useVoiceRecorder.ts`, closest composer/hook tests and localized user text only if necessary. No `runtime.ts` native-host advertisement. Depends on the fixed visibility notification from P1/P3. | Recording and pending permission cleanup pass with fake media/time, retained draft/error and unchanged browser/chat/attachment behavior. Shell hide/show test confirms the same mounted webview. |
| P5: Release integration | Desktop build/RPM metadata and desktop usage/validation documentation; Python/WebUI package inclusion only if required. Same shell owner controls overlapping Tauri metadata. Depends on P2–P4. | Focused and full qualification, installed-backend handshake, real Fedora scenarios, resource measurements and RPM lifecycle checks pass with retained raw evidence. Rollback is rehearsed without touching production data. |

## Deterministic verification and acceptance

For P2/P4, first add tests that fail for the missing behavior, retain the red
output, implement minimally, then refactor and rerun the owning complete modules.
Use real temporary roots and deterministic barriers for races, fake time/media
and controlled subprocesses; no sleeps as race assertions or real provider calls.
For native tray/packaging, the deterministic substitute is a recorded P1 probe
plus release-build before/after scenarios on the same Fedora session.

| Case | Pass condition |
| --- | --- |
| L1: explicit selection | Invalid/missing config or interpreter, unsupported backend/capability and bad `${VAR}` fail before spawn/setup writes; no fallback. All staging data roots are isolated from production. |
| L2: launch races | Concurrent desktop/CLI acquisition and last-client release create at most one process/writer; persistent lifetime is present before spawn/adoption completes. Shutdown-pending acquisition waits/refuses within its deadline. |
| L3: reuse/refusal | Verified matching managed-background gateway reuses the same PID/birth identity and is promoted once. Wrong ports, foreign listener, foreground/service, legacy/unknown identity, shared group and wrong selectors cause no signal or lifetime promotion. |
| L4: readiness | Live PID plus failed health/assets/auth/ID binding is Unavailable. Recovery changes status without a replacement process. A foreign successful HTTP response cannot make the selected backend Running. |
| L5: hide/show | Same webview, chat ID, unsent draft, attachments and stream survive repeated close/Open Chat/icon launches. Hidden gateway still executes one local-fixture scheduled occurrence and one controlled process completion through their existing owners. |
| L6: capture | Close during recording, hold delay and pending `getUserMedia` leaves no live tracks/audio context/timers; output becomes an unsent draft or visible result. Reopen never restarts capture. Browser visibility behavior remains functional. |
| L7: graceful Quit | Active idle/turn/job/continuation fixtures exercise existing cancellation/cleanup and durable outcomes. Desktop waits past state-claim removal until captured process exit; only then exits. No force escalation is invoked. |
| L8: failed/racing Quit | Timeout, pipe loss, PID reuse/replacement and concurrent quit/relaunch retain a responsive error/retry surface. Replacement state/leases survive; no unverified process/group receives a signal. |
| L9: crash/recovery | Kill only controlled fixture host/adapter/gateway processes. Host crash may leave persistent gateway; relaunch attaches once. Interrupted execution stays recoverable/unknown with no desktop replay. |
| S1: credentials/capabilities | Sentinel secrets absent from process argument captures, opener requests, logs/preferences/diagnostics; fragment consumed; HTTP/WS use correct auth/origin. Gateway JS cannot invoke native shell/fs/process/selection/quit commands. |
| S2: navigation | Same-origin route works; foreign/new-window/OAuth fixtures use the constrained opener and cannot replace chat; disallowed schemes/credential-bearing URLs fail. |
| N1: Fedora controls | KDE/Wayland tray menu Open Chat/Retry/Quit, minimize/hide/focus, restart/absence of tray host and session-bus loss have usable controls. Repeated real icon launches create one host/backend. |
| R1: installed package | Run outside checkout with a prepared installed backend and wheel assets; no source-tree, Node, Bun or login-shell PATH dependency. Install/upgrade/uninstall/downgrade preserve selectors/data and do not invoke services or pip. |

Run these future implementation validation commands from the repository root
(not run here). The frontend subshell returns to that root when it exits:

```bash
.venv/bin/python -m pytest -q -p no:cacheprovider tests/desktop tests/gateway/test_runtime.py
.venv/bin/python -m pytest -q -p no:cacheprovider tests/cli/test_webui_support.py tests/cli/test_tui_launcher.py tests/cli/test_gateway_commands.py tests/cli/test_startup_admission_onboard_webui_api.py tests/cli/test_startup_admission_agent_gateway.py
.venv/bin/python -m pytest -q -p no:cacheprovider tests/cli/test_gateway_scheduled_execution.py tests/webui/test_gateway_webui_smoke.py tests/webui/test_gateway_terminal.py tests/tools/test_process_completion.py tests/agent/test_process_completion_scope.py tests/cron/test_gateway_cron.py tests/migration/test_p5_stage1_shutdown.py tests/migration/test_p5_stage3_shutdown.py
(
  cd webui
  bun run test
  bun run build
)
```

`tests/desktop` is a planned directory; the other paths and Bun scripts exist.
Retain existing smoke network/subprocess allowlists and no-model-call fixtures.
For the new desktop subprocess tests exercise the real new admission path;
do not monkeypatch the migration fence as proof that desktop startup works.

Every future Python batch follows [AGENTS.md](../AGENTS.md) and
[the P6 attribution policy](migration/p6-error-attribution.md): freeze full-source
before/after `basedpyright --outputjson` reports, exact source hashes, commands,
exit codes, interpreter/checker versions and dependency inventory. Use identical
dependencies/checker policy between reports; never refresh a baseline to absorb
new signatures or suppress scope. Run the actual comparator from the repository
root:

```bash
.venv/bin/python scripts/check_type_regressions.py "$DESKTOP_QA/before.json" "$DESKTOP_QA/after.json" --baseline-root "$DESKTOP_BEFORE_ROOT" --current-root "$DESKTOP_AFTER_ROOT"
```

These are future evidence paths to allocate explicitly, not existing artifacts.
The comparator must exit zero with **zero introduced signature multiplicities
across all files**, including edited/new files; lower totals alone do not pass.
It does not attest environment/source parity, which must be recorded separately.
Prepare dependencies once using the repository's `uv sync --all-extras --dev`
and channel dependency installer when authorized, then freeze them. The unchanged
strict command remains `uv run --no-sync basedpyright`; preserve the same
environment for JSON captures and final checking. Do not run `ruff format`.

Before claiming release completion require full strict exit zero, applicable
lint/builds and both parts of the repository's qualified local full suite.
Run this block from the repository root.

```bash
COLUMNS=240 .venv/bin/python -m pytest -q -p no:cacheprovider -n 4 --dist loadfile --ignore=tests/cli/test_commands.py -o faulthandler_timeout=90
COLUMNS=240 .venv/bin/python -m pytest -q -p no:cacheprovider tests/cli/test_commands.py
.venv/bin/ruff check nanobot tests conftest.py scripts/import_cron_copy.py
git diff --check
```

Retain skips/failures and clean process-exit evidence; a timed-out runner with
printed passes is not PASS. A bounded batch or this plan does not close P6.
No checker/dependency weakening, new skips or automatic production authorization.

Run the planned Rust commands from `desktop/src-tauri`:
`cargo test --locked`, `cargo clippy --locked --all-targets -- -D warnings`, and
`cargo tauri build --bundles rpm` after selecting/installing a compatible CLI.
Pin compatible stable Tauri 2 crates/plugins and commit `Cargo.lock`; these
commands do not exist as a current repository desktop workflow yet.

On the Fedora release build, record backend and desktop/WebKit process trees
separately: 60-second idle baseline, visible chat, hidden at 30/300 seconds,
restore, and ten hide/show cycles, using the same local transcript/workload.
Collect PSS/RSS (do not double-count shared pages), CPU and show/ready latency
with versions/source hashes. Report actual numbers and per-component deltas;
do not invent a memory target or claim hidden renderer release. No monotonic
process/window/track growth is allowed across cycles. Tauri uses multiple
processes and the system WebKitGTK runtime; measurements must include those
processes. [Process model](https://v2.tauri.app/concept/process-model/),
[webview versions](https://v2.tauri.app/reference/webview-versions/).

## Fedora build, RPM contract and rollout

Target **Fedora 44 x86_64** first; wider Fedora/glibc support needs its own oldest
supported build baseline and tests. Prerequisites are Rust stable/Cargo, a
compatible Tauri 2 CLI, C/C++ build tools/pkg-config, and Fedora development
packages `webkit2gtk4.1-devel`, `openssl-devel`, `libappindicator-gtk3-devel`,
`librsvg2-devel`, `libxdo-devel` (plus their resolved dependencies). Bun is needed
to build/test changed WebUI assets, not to launch an installed app. Resolve
actual package availability before installation; this plan authorizes none.
These names follow the [Tauri Fedora prerequisites](https://v2.tauri.app/start/prerequisites/).

The RPM contains the native host, icons, `.desktop` entry, small status/setup
assets, licenses/notices and a tested backend compatibility manifest. `Exec`
starts only the desktop binary without config secrets. Declare the required
WebKitGTK/GTK/appindicator system runtime libraries, inspect generated RPM
dependencies, and test them on a clean Fedora image; installing runtime libraries
does not install development headers. The already installed backend owns its
Python dependencies and compiled WebUI. Do not bundle a second copy of chat
assets that could drift, or claim a self-contained installer.

Inspect generated content with `rpm -qpl`, requirements with
`rpm -qp --requires`, and scripts with `rpm -qp --scripts` on the actual artifact.
The [Tauri RPM guide](https://v2.tauri.app/distribute/rpm/) documents dependency
metadata, desktop templates, build output and glibc baseline constraints.
No RPM script starts/stops a user's gateway, installs Python dependencies,
modifies config/data, enables autostart or changes a service. Keep current
headless wheel/sdist inclusion and TUI sidecar/release contracts intact.

1. Implement and qualify with disposable configs and local fixture providers.
   Then rehearse only the explicitly selected
   `/home/jaden/.nanobot-staging/config.json` and isolated workspace/session/cron
   data. Verify selectors/root boundaries before any run. Never attach to
   marketing production or alter `~/.nanobot` originals.
2. Build release RPM and backend artifacts from recorded source; validate from
   outside the checkout against the installed compatible backend. Record Fedora,
   Rust/Tauri/WebKit/backend versions and artifact hashes. Install in a disposable
   Fedora user/image and complete L1–R1 and resource measurements before a personal
   staging installation. No marketing workflow is required to prove the shell.
3. Upgrade only after explicit Quit confirms the managed instance stopped.
   Upgrade host/backend deliberately as a tested compatible pair; an incompatible
   or removed interpreter/backend leads to setup/error UI, not an install or
   fallback. RPM upgrade scripts never stop running user processes. Test both
   compatible and incompatible backend changes.
4. Uninstall removes RPM-owned files only and preserves private selector
   preferences, webview profile and all Python user data. Document explicit Quit
   before uninstall; uninstall itself does not silently kill a running host or
   persistent gateway. Browser/CLI remain usable through the installed backend.
5. Rollback: Quit the managed staging instance, reinstall the prior desktop RPM
   and select its previously qualified interpreter/backend. Desktop introduces no
   session/cron schema conversion. Preserve newer durable backend state; never
   restore a stale data snapshot over it as a desktop rollback. If backend schema
   compatibility prevents downgrade, retain the newer backend and use browser/CLI
   until a compatible shell is available.

## Remaining spikes, gates and readiness

All are **pending**, not completed feasibility evidence:

| Spike/gate | Required outcome / response to failure |
| --- | --- |
| Fedora tray and Wayland | Prove real menu activation/focus and reliable tray-host absence/loss detection. If host detection is inconclusive, retain/minimize a reachable window; do not hide the only controls. Direct-click activation remains optional and unadvertised. |
| WebKit and capture | Prove microphone/MediaRecorder formats, secure loopback, fixed notification-before-hide and late-permission cleanup in the release webview. Unsupported capture yields a clear existing recording error; no invisible capture or broad native bridge. |
| Backend and signal ownership | Prove the existing state/ID publication binds authenticated readiness to the selected PID, and that graceful-only stop/replacement races preserve cleanup and unrelated processes. If existing locks cannot enforce it, return a bounded lifecycle correction before dispatching more shell work. |
| Packaging prerequisites | Supply missing toolchain/headers only in an implementation environment with authorization; qualify actual locked versions and RPM runtime dependencies. No build success is claimed from installed WebKit runtime alone. |
| Release qualification | All relevant acceptance cases, unchanged strict zero, full local suite, frontend/native release builds and clean installed-RPM tests pass with exact evidence. Failures block a release claim, not unrelated read-only planning. |

Routine implementation choices are delegated within these boundaries; there is
no arbitrary confirmation gate. Installing dependencies, changing the user's
installed packages/services, publishing/signing with real credentials, production
cutover and external delivery require their actual task authorization; no
destination, key or recipient is invented here. None is required to finish this
planning document.

AC coverage: AC1 → behavior/states and L5–L9/N1; AC2 → ownership/admission and
L1–L4/L7–L9; AC3 → ordered P1–P5 and preserved core/headless boundaries;
AC4 → webview/authentication and S1–S2; AC5 → verification commands, Fedora
scenarios and separate resource measurements; AC6 → prerequisites/RPM/rollout
and pending spikes; AC7 → full-source frozen diagnostic policy and strict/full
suite qualification; AC8 → this sole planning document and explicit evidence
status. No unresolved product decision is needed before P1/P2. Status:
**plan complete; implementation, native feasibility and release qualification
remain pending**.
