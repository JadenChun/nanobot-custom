# Mobile development checkout

Develop the Android client on `codex/mobile-client` in its dedicated worktree.
The initial base is staging commit `7ad57f9aaf163bd0476c2241637925343b1eb928`.
The reviewed [mobile client plan](mobile-client-plan.md) defines the work packages;
P0 is the Android emulator/native transport proof. iOS remains deferred.

This checkout starts with committed staging source plus the mobile planning documents.
It does not include staging's uncommitted fixes. Staging can continue to receive fixes
and run the installed app while mobile work proceeds here.

## Keep development runtime separate

Git worktrees isolate source files, build output and indexes. They share Git history
and branch references; they do not isolate machine services or runtime data.

- Create dependencies in this worktree's own `.venv` and `webui/node_modules`.
  Do not symlink staging's environment or install this checkout into staging's environment.
- Use a separate development config, instance root, agent data/workspace and unused
  loopback ports for later backend tests. Never use the installed staging instance's
  config, sessions, credentials, permissions or lifecycle handles as a test fixture.
- Start with P0's isolated non-agent fixture. Backend development must not start a
  second writer for the existing staging instance.
- Build WebUI/mobile assets inside this checkout. Do not copy experimental bundles
  into the installed staging app or restart/reinstall that app from this branch.
- Tailscale, Serve and system services are machine resources. Any later test setup
  must use explicitly approved, separately owned mappings and restore them afterward.

No dependencies, emulator, test server, service or network exposure are installed or
started by this checkout setup.

## Bring changes together deliberately

Staging updates do not automatically enter this branch. Integrate selected committed
fixes with an explicit merge or cherry-pick, then rerun the affected mobile checks.
Mobile changes enter staging only through a reviewed, deliberate integration after
their package gates pass. Do not switch, reset or stage files in the staging worktree
as part of mobile development.
