# Mobile development checkout

Develop the Android full-owner client on `codex/mobile-client` in its dedicated worktree,
`/home/jaden/.codex/worktrees/mobile-client/nanobot-custom`. Start with the
[task backlog](mobile-client-tasks.md), [implementation guide](mobile-client-implementation.md) and
[architecture](mobile-client-plan.md). M36 freezes the actual desktop feature/API and implementation
baseline before current work. Earlier implementation/test/APK reports are historical and unverified
by this revision; existing code is neither accepted nor rejected here.

The earlier rebase included staging `a60d8c5037479848d985475eb2e1bc81159d1bb7`.
The four-document review baseline is `b05292b0769732ea99f72c0a74f13f222d98030f`.
Record actual clean HEAD, source/dependency/environment manifests and fresh applicability checks
rather than enforcing historical hashes. Earlier `b0291974`, `43462942` and `7ad57f9a` are historical.
This revision runs only documentation/static checks and changes no runtime or app.

## Keep development runtime separate

Worktrees isolate source/build/index state; they share Git history and do not isolate services/data.

- Use this worktree's own `.venv`, `webui/node_modules` and mobile toolchain/caches. Never symlink or
  install this checkout into installed staging's editable environment.
- Use a separately selected development config, instance root, workspace, credentials, sessions and
  unused loopback ports. Installed staging config/data/permissions/lifecycle handles are never fixtures.
- Begin P0 with the isolated non-agent fixture. Later backend and M35 desktop-equivalent tests use an
  isolated real instance with its own canonical data and exactly one gateway/owner authority.
- Build assets here. No experimental bundle is copied into installed staging and no installed app is
  restarted/reinstalled during qualification. M35 is isolated integration, not installed promotion.
- Tailscale/Serve/services are machine resources: use concrete opted-in isolated resources and exact
  teardown. One external origin may have multiple owned data/control mappings; preserve unrelated entries.
  Owner auth/status/refresh/device-control routing must work with the gateway stopped.
- Preserve durable identity/device/revocation/audit/session state. Fixture deletion is allowed only when
  explicitly disposable; rollback never erases installed owner state or blindly restores stale credentials.

No dependencies, emulator, service, config or network exposure is installed or started by this plan edit.

## Integrate and promote deliberately

M10's historical Git sync does not qualify current permission/API/layout behavior. Check committed
ancestry and current seams after M36. Further staging updates enter this branch only at a clean recorded
checkpoint with old/new source/dependency/checker evidence; never copy or stage dirty staging files.
Do not switch/reset/stage the other worktree as part of mobile development.

Current order: M36 inventory → retained foundations and M30/M31 owner backend → M37 directory backend →
shared/native/UI work → M33 safeguards → M35 isolated desktop-equivalent integration → M20–M22 current
automated/physical/signing qualification → M34 complete parity bundle → M38 independent review.
M01–M22 are not an alternative finish. New remote Start/service-manager control, iOS, actual VPS
deployment/model linking/migration remain separately scoped M23–M29.

M39 is a distinct installed promotion after M38 PASS. Before any installed mutation, select the actual
instance and record current/desired server package+commit, bundled WebUI hash, APK/version, protocol/
registry/schema, executable/environment and launcher, config/instance/data paths, supervisor and owned
mapping changes, backups/compatibility and rollback. Review binds that exact manifest. Stop the old
runtime before the reviewed build opens the same data; retain one writer/owner per instance.
Config/routing changes alone do not upgrade an old installed backend. Install the reviewed backend
and matching bundled WebUI through the selected local installation mechanism, then reconcile config/
services/ingress. A changed candidate returns to affected qualification and review rather than bypassing M38.

M39 requires concrete local install/config/service/network opt-in for the selected transition; this plan
does not supply it. M40 verifies phone and desktop use the actual same installation, history, workspace
policy and supported actions, including stopped-gateway control reachability and revocation. With missing
target/permission/phone/signing inputs, keep the installed result PENDING while independent isolated work
continues. No qualification/review depends on promotion, so no cycle is created.

Rollback disables remote ingress before any incompatible downgrade; retain stable installation identity,
devices, revocations, audit and sessions. A previous backend may resume remote access only after protocol
and durable-store compatibility checks; otherwise keep ingress off with local recovery. Remove only owned
mapping/service resources and disposable fixtures, never global Tailscale or selected-instance durable state.

Invoking the taskbook prompt authorizes ready isolated implementation/tests without per-task human stops.
Present concrete external transitions and recovery before requesting genuinely missing opt-in. Merge,
publication and external production changes need their existing explicit authorization; none occurs here.
