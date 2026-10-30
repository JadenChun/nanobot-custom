# Mobile development checkout

Develop the Android client on `codex/mobile-client` in its dedicated worktree.
The latest clean rebase (2026-10-30) includes committed staging base
`43462942b7b040e35dc98e01a46fe6ee23267f37`; the inspected mobile HEAD is
`b02919744d1765e2c7d899355df46092420afce5`. The initial architecture review used
`7ad57f9aaf163bd0476c2241637925343b1eb928`. Record the actual clean start HEAD after
the documentation handoff commit rather than requiring the inspected HEAD verbatim.
Start with the complete [task backlog](mobile-client-tasks.md) and
[per-task implementation steps](mobile-client-implementation.md), alongside the
reviewed [architecture plan](mobile-client-plan.md). P0 proves Android emulator/native
transport; iOS and future VPS/model/migration work remain separately scoped.

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

Staging updates do not automatically enter this branch. At task M10's clean checkpoint,
rebase/update only this mobile branch after the permission/WebUI fixes are committed
and their affected tests pass; record the new source/dependency manifest and rerun
affected mobile checks. Refresh permission/API/layout assumptions before P2/P4 work.
Never cherry-pick or copy uncommitted staging files. P0 and independent contract/store
work can proceed while staging evolves; its live server is not this branch's fixture.
Mobile changes enter staging only through a reviewed, deliberate integration after
their package gates pass. Do not switch, reset or stage files in the staging worktree
as part of mobile development.

When the user invokes the included OpenCode prompt, it authorizes ready Android tasks
in dependency order without per-task approval stops. This planning task only prepares
the handoff. Measured feasibility/dependency gates remain mandatory;
network/service changes need their concrete local opt-in. Review the integrated result
before merge or production activation. See the taskbook for complete evidence and rollback.
