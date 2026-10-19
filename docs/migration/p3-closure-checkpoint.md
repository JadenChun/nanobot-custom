# P3 adapter-contract closure checkpoint

Status as of 2026-09-29: **P3 adapter contracts are accepted.** Provider,
image, authentication, channel, cron, SDK, and API adapters have deterministic
acceptance evidence using synthetic credentials, fake transports, and offline
fixtures. This acceptance includes the dormant end-to-end claimed cron
composition described in [the composition contract](p3-cron-composition-contract.md).
It supersedes the earlier P3 work-log questions about authentication policy,
Telegram empty-`allowFrom` behavior, scheduled approval policy, and claimed
cron composition.

P3 acceptance is a contract checkpoint, **not runtime or deployment
authorization**. The global migration guard remains active. Cron scheduler
wiring and legacy-fresh admission remain closed, and no live provider,
Telegram, credential, cron-state, scheduler, or deployment path was exercised
or enabled.

## Accepted decisions and behavior

- Canonical Nanobot credentials win. Official/legacy credentials import only
  when the canonical credential is absent and no logout tombstone exists;
  writes remain canonical. Logout creates the tombstone and suppresses
  automatic reimport. A conflicting account is never merged: switch or
  reimport must be explicit and generation-checked.
- Credential persistence covers POSIX and native Windows behavior. Native
  Windows tests exist, but could not execute on the macOS acceptance host;
  Windows execution, ACL, and `FileRenameInfoEx` behavior remain P6
  qualification work.
- An enabled Telegram configuration whose effective `allowFrom` is empty
  fails before dependency loading, startup, or hot mutation. Pairing is a
  secondary check and cannot bypass sender authorization. An invalid hot
  update preserves the last valid runtime.
- Unattended cron execution has no implicit or stored human approval and
  persists no pending approval request.
- Dormant claimed cron composition accepts only a bounded canonical text
  result, an authoritative empty explicit-send ledger, and exact Telegram
  receipt transport. Richer results and other transports fail closed and
  require explicit reconciliation where effects are ambiguous.
- Durable callback observation is content-free and exact-once, with no
  automatic retry. Callback failure cannot replay computation and does not
  prevent delivery from proceeding.

## Acceptance evidence

The commands below were separate selections; counts are intentionally not
summed because some selections may overlap.

- Provider/auth/image integrated command: **476 passed, 3 skipped**. The
  skipped cases were Windows-only credential tests. Pinned
  `oauth-cli-kit==0.1.6` came from a disposable external target; no live
  network or credentials were used.
- Telegram manager hot-reload/policy/authorization command: **55 passed**.
- Telegram channel/differential/headless/concurrency/package command:
  **280 passed**. The broader plugin suite is not green in this local
  environment because the declared `setproctitle` dependency is absent.
- Cron silent/canonical/callback/composition/delivery/reconciliation/occurrence
  command: **241 passed**.
- Cron package D/E command: **25 passed**.
- SDK/API command: **72 passed**, with `python-docx` supplied only from a
  disposable external target. This is not installed-dependency or release
  qualification.
- Scoped Ruff over all changed production and test paths, `compileall`, and
  `git diff --check` passed. `basedpyright` was unavailable.

Independent review accepted authentication after two correction rounds,
Telegram after one, and cron after two. Those corrections closed
symlink/Windows credential-publication races, Telegram hot-update/dependency
bypasses, and callback root/record-substitution races.

## Preserved limits and next gates

The full provider collection still has pre-existing stale imports of removed
`_convert_messages` and `_is_reasoning_model`. Local CLI/environment baselines
still lack declared `dulwich` and `setproctitle` dependencies, and broader SDK
attachment required disposable `python-docx`. Eight legacy cron tests continue
to assert destructive old behavior and remain recorded failures, not green
P3 evidence. The dormant callback store intentionally requires POSIX `flock`
and fails closed elsewhere; a cross-platform callback backend is later
qualification work.

P3 completion does not make the candidate deployable. The next gates are P4
durable context/session work (including disposition of existing unaccepted P4
work), P5 integration and activation, P6 reproducible release and
native-platform qualification, P7 live canary, and H1's durable busy inbox.
No P4/P5/P6/P7/H1 gate, live traffic qualification, guard removal, scheduler
activation, or production migration is implied by this checkpoint.
